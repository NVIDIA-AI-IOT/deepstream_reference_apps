/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <glib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "nvdstilerconfig.h"

typedef struct {
    GstElement *pipeline;
    GstElement *streammux;
    GstElement *tiler;
    gboolean no_display;
    guint num_sources;
    /* Keep these alive for the lifetime of the pipeline */
    CustomTile *tiles_arr;
    CustomTileConfig *cfg;
} AppCtx;

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_printerr("End-of-stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err = NULL;
            gchar *debug = NULL;
            gst_message_parse_error(msg, &err, &debug);
            g_printerr("Error: %s\n", err ? err->message : "(unknown)");
            if (debug) g_printerr("Debug details: %s\n", debug);
            g_clear_error(&err);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

static void
cb_newpad(GstElement *decodebin, GstPad *pad, gpointer user_data)
{
    GstCaps *caps = NULL;
    const gchar *name = NULL;
    GstStructure *str = NULL;
    GstBin *source_bin = GST_BIN(user_data);

    caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, NULL);
    if (!caps) return;
    str = gst_caps_get_structure(caps, 0);
    name = gst_structure_get_name(str);

    if (name && g_str_has_prefix(name, "video")) {
        /* Only link NVMM pads */
        GstCapsFeatures *features = gst_caps_get_features(caps, 0);
        if (features && gst_caps_features_contains(features, "memory:NVMM")) {
            GstPad *ghost = gst_element_get_static_pad(GST_ELEMENT(source_bin), "src");
            if (ghost) {
                if (!gst_ghost_pad_set_target(GST_GHOST_PAD(ghost), pad)) {
                    g_printerr("Failed to set ghost pad target\n");
                }
                gst_object_unref(ghost);
            }
        } else {
            g_printerr("Decodebin did not pick NVMM memory\n");
        }
    }
    if (caps) gst_caps_unref(caps);
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data)
{
    if (g_strrstr(name, "decodebin")) {
        g_signal_connect(object, "child-added", G_CALLBACK(decodebin_child_added), user_data);
    }
}

static GstElement *
create_source_bin(guint index, const gchar *uri)
{
    gchar bin_name[32];
    g_snprintf(bin_name, sizeof(bin_name), "source-bin-%02u", index);
    GstElement *bin = gst_bin_new(bin_name);
    if (!bin) return NULL;

    GstElement *uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");
    if (!uri_decode_bin) {
        g_printerr("Unable to create uridecodebin\n");
        gst_object_unref(bin);
        return NULL;
    }
    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);
    g_signal_connect(uri_decode_bin, "pad-added", G_CALLBACK(cb_newpad), bin);
    g_signal_connect(uri_decode_bin, "child-added", G_CALLBACK(decodebin_child_added), bin);

    gst_bin_add(GST_BIN(bin), uri_decode_bin);

    /* Create an initially no-target ghost pad that will be set in cb_newpad */
    GstPad *ghost = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
    if (!ghost) {
        g_printerr("Failed to add ghost pad\n");
        gst_object_unref(bin);
        return NULL;
    }
    gst_element_add_pad(bin, ghost);
    return bin;
}

static void
build_custom_layout(AppCtx *app)
{
    /* Example asymmetric layout similar to the Python sample */
    guint n = app->num_sources;
    if (n == 0) return;

    guint length = n;
    CustomTile *tiles = g_new0(CustomTile, length);

    /* Default: 2x2 style base with one big-left if >=3 */
    if (length >= 1) { tiles[0].sourceId = 0; tiles[0].x = 0.00f; tiles[0].y = 0.00f; tiles[0].width = (n >= 3 ? 0.66f : 1.0f); tiles[0].height = 1.00f; }
    if (length >= 2) { tiles[1].sourceId = 1; tiles[1].x = (n >= 3 ? 0.66f : 0.00f); tiles[1].y = 0.00f; tiles[1].width = (n >= 3 ? 0.34f : 1.0f - tiles[0].x); tiles[1].height = (n >= 3 ? 0.50f : 1.0f); }
    if (length >= 3) { tiles[2].sourceId = 2; tiles[2].x = 0.66f; tiles[2].y = 0.50f; tiles[2].width = 0.34f; tiles[2].height = 0.50f; }
    if (length >= 4) { tiles[3].sourceId = 3; tiles[3].x = 0.33f; tiles[3].y = 0.33f; tiles[3].width = 0.33f; tiles[3].height = 0.33f; }

    app->tiles_arr = tiles;
    app->cfg = g_new0(CustomTileConfig, 1);
    app->cfg->tiles = app->tiles_arr;
    app->cfg->length = length;
}

static gboolean
update_layout_cb(gpointer user_data)
{
    AppCtx *app = (AppCtx *)user_data;
    if (!app || !app->cfg || !app->cfg->tiles || app->cfg->length == 0) return G_SOURCE_REMOVE;

    /* Make the first tile full screen as a demo update */
    app->cfg->tiles[0].x = 0.0f;
    app->cfg->tiles[0].y = 0.0f;
    app->cfg->tiles[0].width = 1.0f;
    app->cfg->tiles[0].height = 1.0f;

    g_object_set(G_OBJECT(app->tiler), "custom-tile-config", (gpointer)app->cfg, NULL);
    g_printerr("Custom layout updated at runtime\n");
    return G_SOURCE_REMOVE; /* one-shot */
}

int
main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    AppCtx app;
	int current_device = -1;
	gboolean enc_hw_support = TRUE;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

	if (prop.integrated) {
      FILE* ptr;
      char device_name[50];
      ptr = fopen("/proc/device-tree/model", "r");

      if(ptr){
        while (fgets(device_name, 50, ptr) != NULL) {
          if (strstr(device_name,"Orin") && (strstr(device_name,"Nano")))
            enc_hw_support = FALSE;
        }
        fclose(ptr);
      }
    }
	
    memset(&app, 0, sizeof(app));

    gst_init(&argc, &argv);

    /* Simple arg handling: program -i uri1 uri2 ... [--no-display] */
    GPtrArray *uris = g_ptr_array_new_with_free_func(g_free);
    for (int i = 1; i < argc; ++i) {
        if (g_strcmp0(argv[i], "-i") == 0 || g_strcmp0(argv[i], "--input") == 0) {
            for (int j = i + 1; j < argc && argv[j][0] != '-'; ++j) {
                g_ptr_array_add(uris, g_strdup(argv[j]));
                i = j;
            }
        } else if (g_strcmp0(argv[i], "--no-display") == 0) {
            app.no_display = TRUE;
        }
    }

    if (uris->len == 0) {
        g_printerr("Usage: %s -i <uri1> [uri2 ...] [--no-display]\n", argv[0]);
        g_ptr_array_unref(uris);
        return -1;
    }

    app.num_sources = uris->len;

    app.pipeline = gst_pipeline_new("ds-custom-tiler");
    if (!app.pipeline) {
        g_printerr("Failed to create pipeline\n");
        g_ptr_array_unref(uris);
        return -1;
    }

    app.streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    if (!app.streammux) {
        g_printerr("Failed to create nvstreammux\n");
        gst_object_unref(app.pipeline);
        g_ptr_array_unref(uris);
        return -1;
    }
    g_object_set(G_OBJECT(app.streammux),
                 "width", 1920,
                 "height", 1080,
                 "batch-size", app.num_sources,
                 "batched-push-timeout", 33000,
                 NULL);

    gst_bin_add(GST_BIN(app.pipeline), app.streammux);

    for (guint i = 0; i < app.num_sources; ++i) {
        GstElement *src_bin = create_source_bin(i, (const gchar *)g_ptr_array_index(uris, i));
        if (!src_bin) {
            g_printerr("Failed to create source bin %u\n", i);
            gst_object_unref(app.pipeline);
            g_ptr_array_unref(uris);
            return -1;
        }
        gst_bin_add(GST_BIN(app.pipeline), src_bin);

        gchar pad_name[32];
        g_snprintf(pad_name, sizeof(pad_name), "sink_%u", i);
        GstPad *sinkpad = gst_element_request_pad_simple(app.streammux, pad_name);
        GstPad *srcpad = gst_element_get_static_pad(src_bin, "src");
        if (!sinkpad || !srcpad) {
            g_printerr("Failed to get pads for source %u\n", i);
            if (sinkpad) gst_object_unref(sinkpad);
            if (srcpad) gst_object_unref(srcpad);
            gst_object_unref(app.pipeline);
            g_ptr_array_unref(uris);
            return -1;
        }
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link source bin to streammux for %u\n", i);
            gst_object_unref(sinkpad);
            gst_object_unref(srcpad);
            gst_object_unref(app.pipeline);
            g_ptr_array_unref(uris);
            return -1;
        }
        gst_object_unref(sinkpad);
        gst_object_unref(srcpad);
    }

    GstElement *queue1 = gst_element_factory_make("queue", "queue1");
    GstElement *queue2 = gst_element_factory_make("queue", "queue2");
    GstElement *queue3 = gst_element_factory_make("queue", "queue3");
    GstElement *queue4 = gst_element_factory_make("queue", "queue4");
    GstElement *queue5 = gst_element_factory_make("queue", "queue5");

    GstElement *pgie = gst_element_factory_make("queue", "primary-inference"); /* placeholder */
    app.tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "convertor");
    GstElement *nvosd = gst_element_factory_make("nvdsosd", "onscreendisplay");
    GstElement *sink = NULL;
	if(app.no_display) {
	  sink = gst_element_factory_make("nvvideoencfilesinkbin", "fakesink");
	} else {
	  if (prop.integrated) {
		sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
	  } else {
#ifdef __aarch64__
        sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
#else
        sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
#endif
	  }
	}

    if (!queue1 || !queue2 || !queue3 || !queue4 || !queue5 || !pgie || !app.tiler || !nvvidconv || !nvosd || !sink) {
        g_printerr("Failed to create one or more elements\n");
        gst_object_unref(app.pipeline);
        g_ptr_array_unref(uris);
        return -1;
    }

    g_object_set(G_OBJECT(nvosd), "process-mode", 0, "display-text", 1, NULL);
    g_object_set(G_OBJECT(app.tiler), "square-seq-grid", FALSE, NULL);
    g_object_set(G_OBJECT(app.tiler), "compute-hw", 1, NULL);
    if (!app.no_display) g_object_set(G_OBJECT(sink), "qos", FALSE, NULL);
    if (app.no_display) {
	  g_object_set(G_OBJECT(sink), "output-file", "/tmp/tile_out.mp4", "enc-type", 1, NULL);
	  if (!enc_hw_support){
        g_object_set(G_OBJECT(sink), "enc-type", 1, NULL);
      }
    }

    /* Initial custom asymmetric layout */
    build_custom_layout(&app);
    if (app.cfg) {
        g_object_set(G_OBJECT(app.tiler), "custom-tile-config", (gpointer)app.cfg, NULL);
	//g_timeout_add_seconds(5, update_layout_cb, &app);
    }

    gst_bin_add_many(GST_BIN(app.pipeline),
                     queue1, pgie, queue2, app.tiler, queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);

    if (!gst_element_link_many(app.streammux, queue1, pgie, queue2, app.tiler, queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
        g_printerr("Failed to link elements\n");
        gst_object_unref(app.pipeline);
        g_ptr_array_unref(uris);
        return -1;
    }

    /* Rows/Columns computed like the Python sample */
    guint rows = (guint)floor(sqrt((double)app.num_sources));
    if (rows == 0) rows = 1;
    guint cols = (guint)ceil(((double)app.num_sources) / (double)rows);
    g_object_set(G_OBJECT(app.tiler), "rows", rows, "columns", cols, "width", 1280, "height", 720, NULL);

    loop = g_main_loop_new(NULL, FALSE);
    GstBus *bus = gst_element_get_bus(app.pipeline);
    gst_bus_add_signal_watch(bus);
    g_signal_connect(G_OBJECT(bus), "message", G_CALLBACK(bus_call), loop);
    gst_object_unref(bus);

    g_print("Now playing...\n");
    for (guint i = 0; i < uris->len; ++i) {
        g_print("%u: %s\n", i, (gchar *)g_ptr_array_index(uris, i));
    }

    gst_element_set_state(app.pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(app.pipeline, GST_STATE_NULL);
    g_main_loop_unref(loop);

    /* Cleanup */
    if (app.tiles_arr) g_free(app.tiles_arr);
    if (app.cfg) g_free(app.cfg);
    gst_object_unref(app.pipeline);
    g_ptr_array_unref(uris);
    return 0;
}

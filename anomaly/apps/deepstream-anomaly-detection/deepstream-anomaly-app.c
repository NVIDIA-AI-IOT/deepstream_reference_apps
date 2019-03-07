/*******************************************************************************
 * MIT License
 *
 * Copyright (C) 2019 NVIDIA CORPORATION
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "gst-nvmessage.h"

/* maximum number of sources that are allowed */
#define MAX_NUM_SRCS 20
/* frame dimension in batched buffer */
#define DEFAULT_VIDEO_WIDTH 1280
#define DEFAULT_VIDEO_HEIGHT 720

/* naming of gst elements */
#define STREAM_MUXER_NAME "streammuxer"
#define TILED_SINK_NAME "tiled-sink"
#define SINK_TEE_NAME "sink-tee"
#define DIRECTION_BIN_NAME_V "optflow-bin-%02d"
#define PRIMARY_MOTION_SINK_PAD_NAME "sink_primary"
#define SECONDARY_MOTION_SINK_PAD_NAME "sink_secondary"
#define SRC_TEE_NAME_V "src-tee-%02d"
#define SOURCE_BIN_NAME "source-bin-"
#define SOURCE_BIN_NAME_V SOURCE_BIN_NAME "%02d"
#define GET_SOURCE_INDEX(name) (atoi(name + strlen(SOURCE_BIN_NAME)))
#define NAME_LENGTH 18

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* use this macro to adjust batch size when current number of active streams changes.
   preferable the nvstreammux plug in can handle it automatically */
#define NEED_BATCH_SIZE_ADJUSTMENT

/* Global application status */
struct
{
    gchar *url_list[MAX_NUM_SRCS]; /*< URL list */
    guint active_streams; /*< number of active streams */
    GMutex lock;          /*< application status mutex */
    GstPipeline* pipeline; /*< application pipeline */
    guint video_width;    /*< width of input video */
    guint video_height;   /*< height of input video */
} app_ctx;

static gchar **input_files = NULL;
static gchar *config_file = NULL;
static gboolean enable_heatmap = FALSE;
static int opticalflow_grid = 2;

gboolean cb_resolution_parse(const gchar* option, const gchar* value, gpointer data, GError **error)
{
    g_print("Get resolution %s\n", value);
    sscanf(value, "%dx%d", &app_ctx.video_width, &app_ctx.video_height);
    return TRUE;
}

GOptionEntry entries[] =
{
    {"input-file", 'i', 0, G_OPTION_ARG_FILENAME_ARRAY, &input_files,
     "Set the input file", NULL},
     {"config-file", 'c', 0, G_OPTION_ARG_FILENAME, &config_file,
      "Set the config file", NULL},
    {"enable-heatmap", 'm', 0, G_OPTION_ARG_NONE, &enable_heatmap,
     "Enable optical flow heat map output", NULL},
    {"opticalflow-grid", 'g', 0, G_OPTION_ARG_INT, &opticalflow_grid,
     "size of the grid that shares one optical flow vector", NULL},
    {"resolution", 'r', 0, G_OPTION_ARG_CALLBACK, cb_resolution_parse,
     "resolution of input video in the format of wxh", NULL},
    {NULL},
};

/**
 * create an optical flow analysis path
 */
static GstElement *create_optical_flow_sink(guint index, gboolean enable_heatmap)
{
    GstElement *bin = NULL;
    GstElement *queue1 = NULL;
    GstElement *optflow = NULL;
    GstElement *direction = NULL;
    GstElement *queue2;
    GstElement *vidconv = NULL;
    GstElement *xsink = NULL;
    GstPad *gstpad = NULL;
    gchar bin_name[NAME_LENGTH] = {};

    g_snprintf(bin_name, NAME_LENGTH, DIRECTION_BIN_NAME_V, index);
    bin = gst_bin_new(bin_name);
    queue1 = gst_element_factory_make("queue", NULL);
    optflow = gst_element_factory_make("dsopticalflow", NULL);
    direction = gst_element_factory_make("dsdirection", NULL);
    if (!bin || !direction || !optflow || !queue1)
    {
        return NULL;
    }

    if (enable_heatmap)
    {
        queue2 = gst_element_factory_make("queue", NULL);
        vidconv = gst_element_factory_make("videoconvert", NULL);
        xsink = gst_element_factory_make("xvimagesink", NULL);
        if (!xsink || !vidconv || !queue2)
        {
            return NULL;
        }
    }

    guint processing_width = app_ctx.video_width / opticalflow_grid;
    guint processing_height = app_ctx.video_height / opticalflow_grid;
    g_object_set(G_OBJECT(optflow),
                 "processing-width", processing_width,
                 "processing-height", processing_height,
                 "unique-id", index,
                 "enable-heatmap", enable_heatmap, NULL);

    gst_bin_add_many(GST_BIN(bin), queue1, optflow, direction, NULL);

    gst_element_link(queue1, optflow);
    gst_element_link_pads(optflow, "src", direction, "optf_sink");

    if (enable_heatmap)
    {
        /* No need to play optical flow heat map in sync mode */
        g_object_set(G_OBJECT(xsink), "sync", FALSE, NULL);
        gst_bin_add_many(GST_BIN(bin), queue2, vidconv, xsink, NULL);
        gst_element_link_many(direction, queue2, vidconv, xsink, NULL);
    }

    /* sink pad to calculate optical flow from decoder output */
    gstpad = gst_element_get_static_pad(queue1, "sink");
    if (!gst_element_add_pad(bin, gst_ghost_pad_new(PRIMARY_MOTION_SINK_PAD_NAME, gstpad)))
    {
        g_printerr("Failed to add primary sink pad in direction bin\n");
        return NULL;
    }
    gst_object_unref(gstpad);

    /* sink pad to extract object information from nvinfer output */
    gstpad = gst_element_get_static_pad(direction, "sink");
    if (!gst_element_add_pad(bin, gst_ghost_pad_new(SECONDARY_MOTION_SINK_PAD_NAME, gstpad)))
    {
        g_printerr("Failed to add ghost pad in direction bin\n");
        return NULL;
    }
    gst_object_unref(gstpad);

    return bin;
}

/* main display path*/
static GstElement *create_tiled_sink(guint n)
{
    GstElement *bin = NULL;
    GstElement *queue = NULL;
    GstElement *tiler = NULL;
    GstElement *vidconv = NULL;
    GstElement *nvosd = NULL;
    GstElement *eglsink = NULL;
    guint tiler_rows, tiler_columns;
    gchar bin_name[NAME_LENGTH] = {};
    gchar pad_name[NAME_LENGTH] = {};

    do
    {
        bin = gst_bin_new(TILED_SINK_NAME);
        if (bin == NULL)
        {
            g_printerr("Failed to create tiled sink\n");
            break;
        }

        queue = gst_element_factory_make("queue", NULL);
        tiler = gst_element_factory_make("nvmultistreamtiler", NULL);
        vidconv = gst_element_factory_make("nvvidconv", NULL);
        nvosd = gst_element_factory_make("nvosd", NULL);
        eglsink = gst_element_factory_make("nveglglessink", NULL);

        if (!queue || !tiler || !vidconv || !nvosd || !eglsink)
        {
            g_printerr("One element could not be created. Exiting.\n");
            break;
        }

        tiler_columns = (guint)sqrt(n);
        tiler_rows = (guint)ceil(1.0 * n / tiler_columns);
        /* we set the osd properties here */
        g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
                     "width", app_ctx.video_width * tiler_columns, "height", app_ctx.video_height * tiler_rows,
                     NULL);

        gst_bin_add_many(GST_BIN(bin), queue, tiler, vidconv, nvosd, eglsink, NULL);
        if (!gst_element_link_many(queue, tiler, vidconv, nvosd, eglsink, NULL))
        {
            g_printerr("Failed to link elements along with tiler");
            break;
        }

        GstPad *gstpad = gst_element_get_static_pad(queue, "sink");
        if (!gst_element_add_pad(bin, gst_ghost_pad_new("sink", gstpad)))
        {
            g_printerr("Failed to add ghost pad in alt sink bin\n");
            g_object_unref(gstpad);
            break;
        }
        gst_object_unref(gstpad);

        /* passed */
        return bin;
    } while (0);

    /* failed : */
    if (queue)
        g_object_unref(queue);
    if (tiler)
        g_object_unref(tiler);
    if (vidconv)
        g_object_unref(vidconv);
    if (nvosd)
        g_object_unref(nvosd);
    if (eglsink)
        g_object_unref(eglsink);

    return bin;
}

/**
 * The function will be invoked when a compatible video pad is created from a source
 * decode bin.
 * Basically it creates a tee and links it to the video pad which has been created, then
 * it links the tee to the streammux on pad #0 and to a newly created optcial flow path on
 * pad #1.
 * The state of the tee and optical flow bin goes to PLAYING after link is done, however
 * the state of streammux bin is not changed.
 */
static void cb_src_newpad(GstElement *src, GstPad *pad, gpointer data)
{
    gchar *src_name = GST_ELEMENT_NAME(src);
    int index = GET_SOURCE_INDEX(src_name);
    gchar *src_pad_name = NULL;
    gchar sink_pad_name[NAME_LENGTH];
    gchar bin_name[NAME_LENGTH];
    gchar pad_name[NAME_LENGTH];
    GstElement *pipeline = GST_ELEMENT_PARENT(src);
    GstElement *src_tee = NULL;
    GstElement *direction = NULL;
    GstElement *sink_tee = NULL;
    GstElement *streammux = NULL;
    GstElement *tiler = NULL;
    GstCaps *caps = gst_pad_query_caps(pad, NULL);
    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    GstCapsFeatures *features = gst_caps_get_features(caps, 0);

    if (strncmp(name, "video", 5))
    {
        g_print("Not a video stream, skipped\n");
        return;
    }

    if (!gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM))
    {
        g_print("Not a NV compatible video stream, skipped\n");
        return;
    }

    src_pad_name = gst_pad_get_name(pad);
    g_print("new compatible video pad %s added on %s\n", src_pad_name, src_name);

    do
    {
        streammux = gst_bin_get_by_name(GST_BIN(pipeline), STREAM_MUXER_NAME);
        if (!streammux)
        {
            g_printerr("Can't find streammux\n");
            break;
        }

        sink_tee = gst_bin_get_by_name(GST_BIN(pipeline), SINK_TEE_NAME);
        if (!sink_tee)
        {
            g_printerr("Can't find sink tee\n");
            break;
        }

        tiler = gst_bin_get_by_name(GST_BIN(pipeline), TILED_SINK_NAME);
        if (!tiler)
        {
            g_printerr("Can't find tiler sink\n");
            break;
        }

        g_snprintf(bin_name, NAME_LENGTH, SRC_TEE_NAME_V, index);
        src_tee = gst_element_factory_make("tee", bin_name);
        direction = create_optical_flow_sink(index, enable_heatmap);
        if (!src_tee || !direction)
        {
            g_printerr("Failed to create tee or direction sink on src %u\n", index);
            break;
        }

        gst_bin_add_many(GST_BIN(pipeline), src_tee, direction, NULL);

        if (!gst_element_link(src, src_tee))
        {
            g_printerr("Failed to link src %u to source tee\n", index);
        }

        /* critical section for pipeline change */
        g_mutex_lock(&app_ctx.lock);
        app_ctx.active_streams++;

        /* src-tee to streammux */
        GstPad *srcpad = NULL;
        GstPad *sinkpad = NULL;
        srcpad = gst_element_get_request_pad(src_tee, "src_0");
        /* request a new numbered pad from streammux bin and set up the link */
        g_snprintf(sink_pad_name, NAME_LENGTH, "sink_%u", index);
        sinkpad = gst_element_get_request_pad(streammux, sink_pad_name);
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link src decode bin %u to streammux \n", index);
        }
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);

#if defined (NEED_BATCH_SIZE_ADJUSTMENT)
        GValue val = G_VALUE_INIT;
        g_value_init(&val, G_TYPE_UINT);
        g_object_get_property(G_OBJECT(streammux), "batch-size", &val);
        if (g_value_get_uint(&val) < app_ctx.active_streams)
        {
            g_object_set(G_OBJECT(streammux), "batch-size", app_ctx.active_streams, NULL);
        }
#endif
        /* src-tee to direction estimation */
        srcpad = gst_element_get_request_pad(src_tee, "src_1");
        sinkpad = gst_element_get_static_pad(direction, PRIMARY_MOTION_SINK_PAD_NAME);
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link source tee to direction sink %u\n", index);
        }
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);

        /* When a new source is added, we have to break the link between sink tee and
           tiled sink and release the pad in sink tee for the reason that we always
           want the tiled sink to be the last output from the sink tee. */
        srcpad = gst_element_get_static_pad(sink_tee, "src_2");
        if (srcpad)
        {
            gst_element_release_request_pad(sink_tee, srcpad);
            gst_object_unref(srcpad);
        }

        g_snprintf(pad_name, NAME_LENGTH, "src_%d", index);
        srcpad = gst_element_get_request_pad(sink_tee, pad_name);
        sinkpad = gst_element_get_static_pad(direction, SECONDARY_MOTION_SINK_PAD_NAME);
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link source tee to direction sink %u\n", index);
        }
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);

        srcpad = gst_element_get_request_pad(sink_tee, "src_2");
        sinkpad = gst_element_get_static_pad(tiler, "sink");
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
        {
            g_printerr("Failed to link sink tee to tiler sink %u\n", index);
        }
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);

        g_mutex_unlock(&app_ctx.lock); /*< critical section exit */

        /* the element goes to PLAYING immediately */
        if (gst_element_set_state(direction, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Failed to set optical flow to playing\n");
        }

        /* the element goes to PLAYING immediately */
        if (gst_element_set_state(src_tee, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Failed to set source tee to playing\n");
        }

    } while (0);

    g_free(src_pad_name);
}

/**
 * create and add a new source decode bin to the pipeline
 */
static GstElement *add_source(GstElement *pipeline, guint index)
{
    GstElement *bin = NULL, *source = NULL, *h264parser = NULL, *decoder = NULL, *tee = NULL;
    gchar bin_name[NAME_LENGTH] = {};
    GstCaps *caps;

    g_snprintf(bin_name, NAME_LENGTH, SOURCE_BIN_NAME_V, index);
    bin = gst_element_factory_make("uridecodebin", bin_name);
    if (!bin)
    {
        g_printerr("One element in source bin could not be created.\n");
        return NULL;
    }
    g_signal_connect(G_OBJECT(bin), "pad-added", G_CALLBACK(cb_src_newpad), NULL);

    /* We set the input filename to the source element */
    g_object_set(G_OBJECT(bin), "uri", app_ctx.url_list[index], NULL);

    gst_bin_add(GST_BIN(pipeline), bin);

    return bin;
}

/**
 * stop and remove the source decode bin, tee and optical flow path
 */
static gboolean remove_source(GstElement *pipeline, guint index)
{
    GstElement *src_bin = NULL;
    GstElement *src_tee = NULL;
    GstElement *sink_tee = NULL;
    GstElement *direction = NULL;
    GstElement *streammux = NULL;
    GstPad *pad = NULL;
    gchar bin_name[NAME_LENGTH] = {};
    gchar pad_name[NAME_LENGTH] = {};
    GstStateChangeReturn state_ret;

    g_snprintf(bin_name, NAME_LENGTH, SOURCE_BIN_NAME_V, index);
    src_bin = gst_bin_get_by_name(GST_BIN(pipeline), bin_name);
    g_snprintf(bin_name, NAME_LENGTH, SRC_TEE_NAME_V, index);
    src_tee = gst_bin_get_by_name(GST_BIN(pipeline), bin_name);
    g_snprintf(bin_name, NAME_LENGTH, DIRECTION_BIN_NAME_V, index);
    direction = gst_bin_get_by_name(GST_BIN(pipeline), bin_name);
    streammux = gst_bin_get_by_name(GST_BIN(pipeline), STREAM_MUXER_NAME);
    sink_tee = gst_bin_get_by_name(GST_BIN(pipeline), SINK_TEE_NAME);
    if (!src_bin || !src_tee || !direction || !streammux || !sink_tee)
    {
        g_printerr("Can't find some element when deleting source\n");
        return FALSE;
    }

    g_print("Removing source %d\n", index);

    /* demolish the optical flow calculation path first */
    g_snprintf(pad_name, NAME_LENGTH, "src_%d", index);
    pad = gst_element_get_static_pad(sink_tee, pad_name);
    gst_element_release_request_pad(sink_tee, pad);
    gst_object_unref(pad);

    pad = gst_element_get_static_pad(src_tee, "src_1");
    gst_element_release_request_pad(src_tee, pad);
    gst_object_unref(pad);

    state_ret = gst_element_set_state(direction, GST_STATE_NULL);
    switch(state_ret)
    {
    case GST_STATE_CHANGE_ASYNC:
        gst_element_get_state(direction, NULL, NULL, GST_CLOCK_TIME_NONE);
    case GST_STATE_CHANGE_SUCCESS:
        gst_element_set_state(src_tee, GST_STATE_NULL);
        state_ret = gst_element_set_state(src_bin, GST_STATE_NULL);
        break;
    case GST_STATE_CHANGE_FAILURE:
        g_print("STATE CHANGE FAILURE\n\n");
        break;
    case GST_STATE_CHANGE_NO_PREROLL:
        g_print("STATE CHANGE NO PREROLL\n\n");
        break;
    }

    switch(state_ret)
    {
    case GST_STATE_CHANGE_ASYNC:
        gst_element_get_state(src_bin, NULL, NULL, GST_CLOCK_TIME_NONE);
    case GST_STATE_CHANGE_SUCCESS:
        g_snprintf(pad_name, NAME_LENGTH, "sink_%u", index);
        pad = gst_element_get_static_pad(streammux, pad_name);
        gst_element_release_request_pad(streammux, pad);
        app_ctx.active_streams--;

#if defined(NEED_BATCH_SIZE_ADJUSTMENT)
        GValue val = G_VALUE_INIT;
        g_value_init(&val, G_TYPE_UINT);
        g_object_get_property(G_OBJECT(streammux), "batch-size", &val);
        if (g_value_get_uint(&val) > app_ctx.active_streams)
        {
            g_object_set(G_OBJECT(streammux), "batch-size", app_ctx.active_streams, NULL);
        }
#endif
        gst_object_unref(pad);
        break;
    }

    gst_bin_remove(GST_BIN(pipeline), src_bin);
    gst_bin_remove(GST_BIN(pipeline), src_tee);
    gst_bin_remove(GST_BIN(pipeline), direction);

    g_print("Source %d removed\n", index);

    return TRUE;
}

static gboolean keyio_cb(GIOChannel *source, GIOCondition condition, gpointer data)
{
    gchar *buf = NULL;
    GError *error = NULL;
    GstElement *pipeline = (GstElement *)data;
    gsize len;

    if (condition == G_IO_IN)
    {
        g_io_channel_read_line(source, &buf, &len, NULL, &error);
        if (buf == NULL)
        {
            g_printerr("Failed to read line\n");
            return FALSE;
        }

        guint index = (atoi(buf));
        g_free(buf);
        if (index >= 0 && index < MAX_NUM_SRCS)
        {
            gchar bin_name[NAME_LENGTH] = {};
            GstElement *src_bin = NULL;

            g_snprintf(bin_name, NAME_LENGTH, SOURCE_BIN_NAME_V, index);
            g_print("toggle %s in pipeline %s\n", bin_name, GST_ELEMENT_NAME(pipeline));

            src_bin = gst_bin_get_by_name(GST_BIN(pipeline), bin_name);
            if (src_bin == NULL)
            {
                src_bin = add_source(pipeline, index);
                if (gst_element_set_state(src_bin, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
                {
                    g_printerr("Failed to set src decode bin to playing\n");
                    return FALSE;
                }
            }
            else
            {
                remove_source(pipeline, index);
            }
        }
    }

    return TRUE;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
                   GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    case GST_MESSAGE_ELEMENT:
    {
        if (gst_nvmessage_is_stream_eos(msg))
        {
            guint stream_id;
            if (gst_nvmessage_parse_stream_eos(msg, &stream_id))
            {
                g_print("Got EOS from stream %d\n", stream_id);
                remove_source(GST_ELEMENT(app_ctx.pipeline), stream_id);
            }
        }
    }
    default:
        break;
    }
    return TRUE;
}
/**
 * Build the original pipeline with the inference path
 */
static GstPipeline *build_pipeline(int n)
{
    GstElement *pipeline = NULL;
    GstElement *streammux = NULL;
    GstElement *pgie = NULL;
    GstElement *tee = NULL;
    GstElement *tiler = NULL;
    GstElement *queue = NULL;
    gchar bin_name[NAME_LENGTH] = {};
    gchar pad_name[NAME_LENGTH] = {};

    do
    {
        pipeline = gst_pipeline_new("ds-pipeline");
        if (pipeline == NULL)
        {
            g_printerr("Failed to create pipeline\n");
            break;
        }

        streammux = gst_element_factory_make("nvstreammux", STREAM_MUXER_NAME);
        pgie = gst_element_factory_make("nvinfer", NULL);
        queue = gst_element_factory_make("queue", NULL);
        tee = gst_element_factory_make("tee", SINK_TEE_NAME);
        tiler = create_tiled_sink(n);
        if (!streammux || !pgie || !tee || !tiler)
        {
            g_printerr("Failed to create elements during building pipeline\n");
            break;
        }

        g_object_set(G_OBJECT(streammux), "width", app_ctx.video_width, "height", app_ctx.video_height, "batch-size", n, NULL);
        g_object_set(G_OBJECT(pgie), "config-file-path", config_file, NULL);
        g_object_set(G_OBJECT(pgie), "batch-size", n, NULL);

        gst_bin_add_many(GST_BIN(pipeline), streammux, pgie, queue, tee, tiler, NULL);
        if (!gst_element_link_many(streammux, queue, pgie, tee, NULL))
        {
            g_printerr("Failed to link elements along with stream muxer");
            break;
        }

        if (gst_element_set_state(tiler, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE ||
            gst_element_set_state(tee, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE ||
            gst_element_set_state(pgie, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE ||
            gst_element_set_state(streammux, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Failed to set streammux to be playing");
            break;
        }

        for (int i = 0; i < n; i++)
        {
            GstElement *src_bin = add_source(pipeline, i);
            if (src_bin == NULL)
            {
                g_printerr("Failed to create src decode bin %d\n", i);
            }
        }

        /* passed */
        return GST_PIPELINE(pipeline);
    } while (0);

    /* failed : */
    if (pipeline)
        g_object_unref(pipeline);
    if (streammux)
        g_object_unref(streammux);
    if (pgie)
        g_object_unref(pgie);
    if (tee)
        g_object_unref(tee);
    if (tiler)
        g_object_unref(tiler);

    return NULL;
}

int main(int argc, char *argv[])
{
    GstBus *bus = NULL;
    GMainLoop *loop = NULL;
    GIOChannel *io = NULL;
    GOptionContext *ctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;

    /* application context initialization */
    g_mutex_init(&app_ctx.lock);
    app_ctx.active_streams = 0;
    app_ctx.pipeline = NULL;
    app_ctx.url_list[0] = NULL;
    app_ctx.video_width = DEFAULT_VIDEO_WIDTH;
    app_ctx.video_height = DEFAULT_VIDEO_HEIGHT;

    ctx = g_option_context_new("Nvidia DeepStream Demo");
    g_option_context_add_main_entries(ctx, entries, NULL);
    g_option_context_add_group(ctx, gst_init_get_option_group());

    if (!g_option_context_parse(ctx, &argc, &argv, &error))
    {
        return -1;
    }

    guint nfiles = g_strv_length (input_files);
    if (nfiles == 0)
    {
        g_print("At least one input file is required\n");
        return -2;
    }

    nfiles = nfiles >= MAX_NUM_SRCS ? MAX_NUM_SRCS : nfiles;
    for (int i = 0; i < nfiles; i++)
    {
        app_ctx.url_list[i] = g_strdup(input_files[i]);
    }
    app_ctx.url_list[nfiles] = NULL;

    io = g_io_channel_unix_new(0);
    g_io_channel_set_encoding(io, NULL, NULL);
    loop = g_main_loop_new(NULL, FALSE);

    app_ctx.pipeline = build_pipeline(nfiles);
    if (app_ctx.pipeline == NULL)
    {
        g_printerr("Failed to build pipeline\n");
        return -1;
    }

    bus = gst_pipeline_get_bus(app_ctx.pipeline);
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_io_add_watch(io, G_IO_IN, keyio_cb, (gpointer)app_ctx.pipeline);

    gst_element_set_state(GST_ELEMENT(app_ctx.pipeline), GST_STATE_PAUSED);

    if (gst_element_set_state(GST_ELEMENT(app_ctx.pipeline), GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
    {
        g_printerr("Failed to set pipeline to playing\n");
        return -1;
    }

    g_main_loop_run(loop);

    gst_element_set_state(GST_ELEMENT(app_ctx.pipeline), GST_STATE_NULL);
    gst_object_unref(app_ctx.pipeline);
    app_ctx.pipeline = NULL;
    g_main_loop_unref(loop);
    g_io_channel_unref(io);

    if (ctx)
    {
        g_option_context_free(ctx);
    }

    g_mutex_clear(&app_ctx.lock);

    gst_deinit();

    return 0;
}

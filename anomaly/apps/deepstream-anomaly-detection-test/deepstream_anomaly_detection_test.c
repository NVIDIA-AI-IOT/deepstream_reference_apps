/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 33 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33000

#define TILED_OUTPUT_WIDTH_INFER 1280
#define TILED_OUTPUT_HEIGHT_INFER 720

#define TILED_OUTPUT_WIDTH_OF 640
#define TILED_OUTPUT_HEIGHT_OF 360

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

#define PGIE_CONFIG_FILE  "dsanomaly_pgie_config.txt"
#define PGIE_NVINFERSERVER_CONFIG_FILE "dsanomaly_pgie_nvinferserver_config.txt"

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing NvBufSurface. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvv4l2decoder. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <uri1> [uri2] ... [uriN]\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <uri1> [uri2] ... [uriN]\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *streammux_queue = NULL,
      *sink_of = NULL, *pgie_queue = NULL, *dsdirection_queue = NULL,
      *nvvidconv_queue = NULL, *nvosd_queue = NULL, *tiler_infer_queue = NULL,
      *tiler_of = NULL, *nvof = NULL, *nvofvisual = NULL, *dsdirection = NULL,
      *of_queue = NULL, *ofvisual_queue = NULL, *sink_infer = NULL,
      *tiler_infer = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL, *tee = NULL, *of_branch_queue = NULL, *infer_branch_queue =
      NULL;

  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean is_nvinfer_server = FALSE;

  GstPad *tee_of_pad, *tee_infer_pad;
  GstPad *queue_of_pad, *queue_infer_pad;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    usage(argv[0]);
    return -1;
  }

  if (argc >=2 && !strcmp("-t", argv[1])) {
    if (!strcmp("inferserver", argv[2])) {
      is_nvinfer_server = TRUE;
    } else {
      usage(argv[0]);
      return -1;
    }
    g_print ("Using nvinferserver as the inference plugin\n");
  }

  if (is_nvinfer_server) {
    num_sources = argc - 3;
  } else {
    num_sources = argc - 1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("anomaly-detection-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  streammux_queue = gst_element_factory_make ("queue", "streammux-queue");

  if (!pipeline || !streammux || !streammux_queue) {
    g_printerr ("(Line=%d) One element could not be created. Exiting.\n",
        __LINE__);
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    GstElement *source_bin;
    gchar pad_name[16] = { };
    if (is_nvinfer_server) {
      source_bin = create_source_bin (i, argv[i + 3]);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Create a tee for two sinks. */
  tee = gst_element_factory_make ("tee", "tee");

  /* Use nvinfer/nvinferserver to infer on batched frame. */
  pgie = gst_element_factory_make (
        is_nvinfer_server ? NVINFERSERVER_PLUGIN : NVINFER_PLUGIN,
        "primary-nvinference-engine");
  pgie_queue = gst_element_factory_make ("queue", "nvinfer-queue");

  /* For Optical Flow output */
  tiler_of = gst_element_factory_make ("nvmultistreamtiler", "nvtiler-of");
  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler_infer =
      gst_element_factory_make ("nvmultistreamtiler", "nvtiler-infer");
  tiler_infer_queue =
      gst_element_factory_make ("queue", "nvtiler-infer-queue");

  /* create nv optical flow element */
  nvof = gst_element_factory_make ("nvof", "nvopticalflow");

  /* create nv ds direction element */
  dsdirection = gst_element_factory_make ("dsdirection", "dsdirection");
  dsdirection_queue = gst_element_factory_make ("queue", "dsdirection-queue");

  /* create nv optical flow visualisation element */
  nvofvisual = gst_element_factory_make ("nvofvisual", "nvopticalflowvisual");

  /* create queue element */
  of_queue = gst_element_factory_make ("queue", "q_after_of");

  /* create queue element */
  ofvisual_queue = gst_element_factory_make ("queue", "q_after_ofvisual");

  /* create queue element */
  of_branch_queue = gst_element_factory_make ("queue", "q_of");

  /* create queue element */
  infer_branch_queue = gst_element_factory_make ("queue", "q_infer");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  nvvidconv_queue = gst_element_factory_make ("queue", "nvvideoconvert-queue");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  nvosd_queue = gst_element_factory_make ("queue", "nvdsosd-queue");

  /* Finally render the osd output */
  if(prop.integrated) {
    sink_of = gst_element_factory_make ("nv3dsink", "nv3dsink-of");
    sink_infer =
        gst_element_factory_make ("nv3dsink", "nv3dsink-infer");

  } else {
    sink_of = gst_element_factory_make ("nveglglessink", "nvelgglessink-of");
    sink_infer =
        gst_element_factory_make ("nveglglessink", "nvelgglessink-infer");
  }

  if (!tee) {
    g_printerr ("Tee could not be created. Exiting.\n");
    return -1;
  }

  if (!nvof || !dsdirection || !nvofvisual || !tiler_of || !sink_of) {
    g_printerr ("One OF element could not be created. Exiting.\n");
    return -1;
  }

  if (!pgie || !tiler_infer || !nvvidconv || !nvosd || !sink_infer) {
    g_printerr ("One Infer element could not be created. Exiting.\n");
    return -1;
  }

  if (!pgie_queue || !tiler_infer_queue || !nvvidconv_queue || !nvosd_queue) {
    g_printerr ("One Queue element could not be created. Exiting.\n");
    return -1;
  }

  /* We set the sync value of both sink elements */
  g_object_set (G_OBJECT (sink_of), "sync", 1, NULL);
  g_object_set (G_OBJECT (sink_infer), "sync", 1, NULL);

  g_object_set (G_OBJECT (streammux), "sync-inputs", TRUE, NULL);
  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer/nvinferserver element using the config file. */
  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_NVINFERSERVER_CONFIG_FILE, NULL);
  } else {
    g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  }

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler_of), "rows", tiler_rows, "columns",
      tiler_columns, "width", TILED_OUTPUT_WIDTH_OF, "height",
      TILED_OUTPUT_HEIGHT_OF, NULL);
  g_object_set (G_OBJECT (tiler_infer), "rows", tiler_rows, "columns",
      tiler_columns, "width", TILED_OUTPUT_WIDTH_INFER, "height",
      TILED_OUTPUT_HEIGHT_INFER, NULL);

  /* We set the sink properties here */
  g_object_set (G_OBJECT (sink_of), "window-x", 0, "window-y", 0, NULL);
  g_object_set (G_OBJECT (sink_infer), "window-x", TILED_OUTPUT_WIDTH_OF,
      "window-y", TILED_OUTPUT_HEIGHT_OF, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), streammux_queue, pgie, pgie_queue, 
      nvof, of_queue, dsdirection, dsdirection_queue, tee,
      of_branch_queue, nvofvisual, ofvisual_queue, tiler_of, sink_of,
      infer_branch_queue, tiler_infer, tiler_infer_queue, nvvidconv,
      nvvidconv_queue, nvosd, nvosd_queue, sink_infer, NULL);


  if ((!gst_element_link_many (streammux, streammux_queue, pgie, pgie_queue,
              nvof, of_queue, dsdirection, dsdirection_queue, tee, NULL))
      || (!gst_element_link_many (of_branch_queue, nvofvisual, ofvisual_queue,
              tiler_of, NULL))
      || (!gst_element_link_many (infer_branch_queue, tiler_infer,
              tiler_infer_queue, nvvidconv, nvvidconv_queue, nvosd,
        nvosd_queue, NULL)) ||
      (!gst_element_link_many (tiler_of, sink_of, NULL)) ||
      (!gst_element_link_many (nvosd_queue, sink_infer, NULL))) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }


  /* Manually link the Tee, which has "Request" pads */
  tee_of_pad = gst_element_get_request_pad (tee, "src_%u");
  g_print ("Obtained request pad %s for OF branch.\n",
      gst_pad_get_name (tee_of_pad));
  queue_of_pad = gst_element_get_static_pad (of_branch_queue, "sink");
  tee_infer_pad = gst_element_get_request_pad (tee, "src_%u");
  g_print ("Obtained request pad %s for infer branch.\n",
      gst_pad_get_name (tee_infer_pad));
  queue_infer_pad = gst_element_get_static_pad (infer_branch_queue, "sink");
  if (gst_pad_link (tee_of_pad, queue_of_pad) != GST_PAD_LINK_OK ||
      gst_pad_link (tee_infer_pad, queue_infer_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  gst_object_unref (queue_of_pad);
  gst_object_unref (queue_infer_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing...\n");

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline),
      GST_DEBUG_GRAPH_SHOW_ALL, "nvof_test_playing");

  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  gst_deinit ();
  return 0;
}

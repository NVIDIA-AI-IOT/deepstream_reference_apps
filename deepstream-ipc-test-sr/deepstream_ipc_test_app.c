/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"

#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#define MAX_SOURCE_BINS 8

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 224
#define MUXER_OUTPUT_HEIGHT 224

/* width and height of model output tensor*/
#define MODEL_OUTPUT_WIDTH 672
#define MODEL_OUTPUT_HEIGHT 672

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

typedef struct
{
  gchar *uri;
  gchar *socket_path;
  guint bus_id;
  GstElement *pipeline;
} NvIpcServerPipeline;

typedef struct
{
  gchar *socket_path[MAX_SOURCE_BINS];
  guint bus_id;
  GstElement *pipeline;
} NvIpcClientPipeline;

typedef struct
{
  GMainLoop *loop;
  NvIpcServerPipeline ipcserver[MAX_SOURCE_BINS];
  NvIpcClientPipeline ipcclient;
} AppCtx;

#define FPS_INTERVAL 300

static AppCtx gAppCtx = {0};
static guint cintr = FALSE;
static gboolean g_perf_mode = FALSE;

static gdouble get_current_timestamp()
{
  struct timeval t1;
  double elapsed_time = 0;
  gettimeofday(&t1, NULL);
  elapsed_time = (t1.tv_sec) * 1000.0;
  elapsed_time += (t1.tv_usec) / 1000.0;
  return elapsed_time;
}

static GstPadProbeReturn
server_sink_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  if (g_perf_mode) {
    GstBuffer *buf = (GstBuffer *) info->data;
    if (gst_buffer_is_writable (buf)) {
      GstCaps *caps = gst_caps_new_simple ("video/x-raw",
          "server_send_time", G_TYPE_DOUBLE, get_current_timestamp(), NULL);
      gst_buffer_add_reference_timestamp_meta (buf, caps, 0, 0);
      gst_caps_unref(caps);
      // g_print ("server_send_time %.3f %s\n", get_current_timestamp(), gst_caps_to_string (caps));
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
client_source_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  if (g_perf_mode) {
    GstBuffer *buf = (GstBuffer *) info->data;
    GstReferenceTimestampMeta *meta =
      gst_buffer_get_reference_timestamp_meta (buf, NULL);
    if (meta == NULL) {
      // g_print ("%s: no reference timestamp meta\n", __FUNCTION__);
      return GST_PAD_PROBE_OK;
    }
    GstCaps *caps = meta->reference;
    double server_send_time = 0.0;
    const GstStructure *str = gst_caps_get_structure (caps, 0);
    gst_structure_get_double (str, "server_send_time", &server_send_time);
    g_print ("IPC latency %.3f\n", get_current_timestamp() - server_send_time);
  }
  return GST_PAD_PROBE_OK;
}

/* client_sgie_src_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
client_sgie_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
    l_frame = l_frame->next) {
      num_rects = vehicle_count = person_count = 0;
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);
          if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
            vehicle_count++;
            num_rects++;
          }
          if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
            person_count++;
            num_rects++;
          }
      }
      g_print ("Frame Number = %d Number of objects = %d "
        "Vehicle Count = %d Person Count = %d\n",
        frame_meta->frame_num, num_rects, vehicle_count, person_count);
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
client_osd_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static double start_time = 0.0f;
  static int frame_count = 0;
  if (frame_count == 0) {
    start_time = get_current_timestamp();
  } else if (frame_count % FPS_INTERVAL == 0) {
    int num_sources = *(guint *) u_data;
    double current_time = get_current_timestamp();
    double fps = frame_count / (current_time - start_time) * 1000.0;
    g_print ("AVG FPS (num_sources %d * %.3f): %.3f\n",
              num_sources, fps, num_sources * fps);
  }
  frame_count++;
  return GST_PAD_PROBE_OK;
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler(int signum) {
    struct sigaction action;

    g_print("User Interrupted.. \n");

    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_DFL;

    sigaction(SIGINT, &action, NULL);

    cintr = TRUE;
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (cintr) {
    cintr = FALSE;
    g_main_loop_quit (gAppCtx.loop);
    return FALSE;
  }
  return TRUE;
}

/*
* Function to install custom handler for program interrupt signal.
*/
static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

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
    case GST_MESSAGE_STATE_CHANGED:
    {
      GstState oldstate, newstate;
      gst_message_parse_state_changed (msg, &oldstate, &newstate, NULL);
        switch (newstate) {
          case GST_STATE_PLAYING:
            //g_print ("Pipeline running\n");
            break;
          case GST_STATE_PAUSED:
            if (oldstate == GST_STATE_PLAYING) {
              //g_print ("Pipeline paused\n");
            }
            break;
          case GST_STATE_READY:
            if (oldstate == GST_STATE_NULL) {
              //g_print ("Pipeline ready\n");
            } else {
              //g_print ("Pipeline stopped\n");
            }
            break;
          case GST_STATE_NULL:
            //g_print ("Pipeline Null\n");
            g_main_loop_quit (loop);
            return FALSE;
            break;
          default:
            break;
        }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

/* delete the pipeline */
void
destroy_pipeline(AppCtx* appCtx) {
  for(int i = 0; i < MAX_SOURCE_BINS; i++){
    if(appCtx->ipcserver[i].pipeline) {
      gst_element_set_state (appCtx->ipcserver[i].pipeline, GST_STATE_NULL);
      gst_object_unref (GST_OBJECT (appCtx->ipcserver[i].pipeline));
      g_source_remove (appCtx->ipcserver[i].bus_id);
      g_print("server is closed uri: %s path: %s\n",
        appCtx->ipcserver[i].uri, appCtx->ipcserver[i].socket_path);
      g_free(appCtx->ipcserver[i].uri);
      g_free(appCtx->ipcserver[i].socket_path);
    }
  }
  if(appCtx->ipcclient.pipeline) {
    GstBus *bus = NULL;
    bus = gst_pipeline_get_bus (GST_PIPELINE (appCtx->ipcclient.pipeline));
    while (TRUE) {
      GstMessage *message = gst_bus_pop (bus);
      if (message == NULL)
        break;
      else if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR)
        bus_call (bus, message, appCtx->loop);
      else
        gst_message_unref (message);
    }
    gst_object_unref (bus);
    gst_element_set_state (appCtx->ipcclient.pipeline, GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (appCtx->ipcclient.pipeline));
    g_source_remove (appCtx->ipcclient.bus_id);
    for(int i = 0; i < MAX_SOURCE_BINS; i++){
      if(appCtx->ipcclient.socket_path[i]) {
        g_print("client is closed path: %s\n",  appCtx->ipcclient.socket_path[i]);
        g_free(appCtx->ipcclient.socket_path[i]);
      }
    }
  }
  g_main_loop_unref (appCtx->loop);
  g_print("destroy_pipeline end\n");
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
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
  if (g_strrstr (name, "source") == name) {
    g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { 0 };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("nvurisrcbin", NULL);
  g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
  g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);

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

static int
create_client_pipeline (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *sgie = NULL,
      *sr_conv = NULL, *sr_capsfilter = NULL, *sr_video_template = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL;
  GstCaps *caps = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *src_pad = NULL;
  guint i = 0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gchar tmp_buf[256] = {0};
  gint  tmp_buf_len = 255;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  gAppCtx.loop = loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  gAppCtx.ipcclient.pipeline = pipeline = gst_pipeline_new ("ipc-client-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  num_sources = argc - 2;
  for (i = 0; i < num_sources; i++) {
    GstElement *source = NULL, *caps_filter = NULL, *queue = NULL;
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { 0 };

    const char *socket_path = argv[i + 2];
    gAppCtx.ipcclient.socket_path[i] = strdup(socket_path);
    g_print("client is connected path: %s\n", gAppCtx.ipcclient.socket_path[i]);

    g_snprintf (tmp_buf, tmp_buf_len, "nvunixfdsrc_%u", i);
    source = gst_element_factory_make ("nvunixfdsrc", tmp_buf);
    gst_bin_add (GST_BIN (pipeline), source);
    g_object_set (G_OBJECT(source), "socket-path", gAppCtx.ipcclient.socket_path[i],
      "buffer_timestamp_copy", TRUE, NULL);
    if (g_perf_mode) {
      g_object_set (G_OBJECT(source), "meta-deserialization-lib", 
        "latency_serialization/liblatency_serialization.so", NULL);
    }

    g_snprintf (tmp_buf, tmp_buf_len, "capsfilter_src_%u", i);
    caps_filter = gst_element_factory_make ("capsfilter", NULL);
    if (!caps_filter) {
      g_printerr ("Failed to create caps_filter. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), caps_filter);

    caps = gst_caps_from_string ("video/x-raw(memory:NVMM),format=NV12");
    g_object_set (G_OBJECT(caps_filter), "caps", caps, NULL);
    gst_caps_unref (caps);

    g_snprintf (tmp_buf, tmp_buf_len, "queue_src_%u", i);
    queue = gst_element_factory_make ("queue", tmp_buf);
    if (!queue) {
      g_printerr ("Failed to create queue. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), queue);
    /* link the elements together */
    if (!gst_element_link_many (source, caps_filter, queue, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    src_pad = gst_element_get_static_pad (source, "src");
    if (!src_pad)
      g_print ("Unable to get src pad\n");
    else
      gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_BUFFER,
          client_source_src_pad_buffer_probe, NULL, NULL);
    gst_object_unref (src_pad);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (queue, "src");
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

  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  sgie = gst_element_factory_make ("nvinfer", "second-nvinference-engine");

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");
  sr_conv = gst_element_factory_make ("nvvideoconvert", "sr_conv");
  sr_capsfilter = gst_element_factory_make ("capsfilter", "sr_capsfilter");
  g_snprintf (tmp_buf, tmp_buf_len, "video/x-raw(memory:NVMM),format=NV12, width=%d, height=%d",
    MODEL_OUTPUT_WIDTH, MODEL_OUTPUT_HEIGHT);
  caps = gst_caps_from_string (tmp_buf);
  g_object_set (G_OBJECT(sr_capsfilter), "caps", caps, NULL);
  gst_caps_unref (caps);
  sr_video_template = gst_element_factory_make("nvdsvideotemplate", "nvdsvideotemplate");
  g_object_set(G_OBJECT(sr_video_template), "customlib-name", "./video_template_impl/libnvds_vt_impl.so", NULL);
  g_object_set(G_OBJECT(sr_video_template), "customlib-props", "config-file:config_videotemplate.yml", NULL);


  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (g_perf_mode) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else if(prop.integrated) {
    sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!pgie || !sgie || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "dsipctest_pgie_config.yml", "output-tensor-meta", TRUE, NULL);
  g_object_set (G_OBJECT (sgie),
      "config-file-path", "dsipctest_sgie_config.yml", NULL);
  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    g_object_set (G_OBJECT (sgie), "batch-size", num_sources, NULL);
  }

#ifdef PLATFORM_TEGRA
  /* for NvBufSurfaceMap in nvvideotemplate */
  g_object_set (G_OBJECT(sr_conv), "nvbuf-memory-type", 2, NULL);
  g_object_set (G_OBJECT(sr_conv), "compute-hw", 1, NULL);
  g_object_set (G_OBJECT(tiler), "compute-hw", 1, NULL);
#endif

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set (G_OBJECT (sink), "qos", 0, NULL);
  // g_object_set (G_OBJECT (sink), "sync", FALSE, NULL);
  g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  gAppCtx.ipcclient.bus_id = bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), queue1, pgie,
      sr_conv, sr_capsfilter, sr_video_template, queue2, sgie, tiler,
      queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);

  /* link the elements together */
  if (!gst_element_link_many (streammux, queue1, pgie, sr_conv, sr_capsfilter, sr_video_template,
    queue2, sgie, tiler, queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  src_pad = gst_element_get_static_pad (sgie, "src");
  if (!src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        client_sgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (src_pad);

  src_pad = gst_element_get_static_pad (nvosd, "src");
  if (!src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        client_osd_src_pad_buffer_probe, (gpointer)&num_sources, NULL);
  gst_object_unref (src_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i + 2]);
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  g_print ("Deleting pipeline\n");
  destroy_pipeline(&gAppCtx);
  return 0;
}

static int
create_server_pipeline (int argc, char *argv[])
{
  guint i =0, num_sources = 0;
  GstPad *sink_pad = NULL;
  num_sources = (argc - 2)/2;
  GMainLoop *loop = NULL;
  gAppCtx.loop = loop = g_main_loop_new (NULL, FALSE);

  for (i = 0; i < num_sources; i++) {
    GstElement *pipeline = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstElement *source_bin=NULL, *queue, *sink= NULL;

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    gAppCtx.ipcserver[i].pipeline = pipeline = gst_pipeline_new ("ipc-server-pipeline");
    if (!pipeline) {
      g_printerr ("Failed to create pipeline. Exiting.\n");
      return -1;
    }

    source_bin = create_source_bin (i, argv[(i*2) + 2]);
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), source_bin);
    gAppCtx.ipcserver[i].uri = strdup(argv[(i*2) + 2]);

    queue = gst_element_factory_make ("queue", NULL);
    if (!queue) {
      g_printerr ("Failed to create queue. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), queue);

    sink = gst_element_factory_make ("nvunixfdsink", NULL);
    if (!sink) {
      g_printerr ("Failed to create nvunixfdsink. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), sink);

    gAppCtx.ipcserver[i].socket_path = strdup(argv[(i*2) + 3]);
    g_print("server is started uri: %s path: %s\n",
      gAppCtx.ipcserver[i].uri, gAppCtx.ipcserver[i].socket_path);
    g_object_set (G_OBJECT(sink), "socket-path", gAppCtx.ipcserver[i].socket_path, 
      "buffer_timestamp_copy", TRUE, NULL);
    if (g_perf_mode) {
      g_object_set (G_OBJECT (sink), "sync", FALSE,
        "meta-serialization-lib", "latency_serialization/liblatency_serialization.so", NULL);
    }

    /* link the elements together */
    if (!gst_element_link_many (source_bin, queue, sink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    sink_pad = gst_element_get_static_pad (sink, "sink");
    if (!sink_pad)
      g_print ("Unable to get sink pad\n");
    else
      gst_pad_add_probe (sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        server_sink_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (sink_pad);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    gAppCtx.ipcserver[i].bus_id = bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    gst_element_set_state (gAppCtx.ipcserver[i].pipeline, GST_STATE_PLAYING);
  }

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  g_print ("Deleting pipeline\n");
  destroy_pipeline(&gAppCtx);
  return 0;
}

int
main (int argc, char *argv[])
{
  int ret = 0;

  /* Check input arguments */
  if (argc < 3) {
    g_printerr ("Usage: %s <server/s> <url> <domain_socket_path>\n", argv[0]);
    g_printerr ("OR: %s <client/c> <domain_socket_path>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);

    /* setup signal handler */
  _intr_setup();
  g_timeout_add(400, check_for_interrupt, NULL);

  g_perf_mode = g_getenv("IPC_SR_PERF_MODE") &&
        !g_strcmp0(g_getenv("IPC_SR_PERF_MODE"), "1");
  g_print ("g_perf_mode: %d\n", g_perf_mode);

  if (strcmp(argv[1], "client") == 0 || strcmp(argv[1], "c") == 0) {
    signal(SIGPIPE, SIG_IGN);
    ret = create_client_pipeline(argc, argv);
  } else if (strcmp(argv[1], "server") == 0 || strcmp(argv[1], "s") == 0) {
    signal(SIGPIPE, SIG_IGN);
    ret = create_server_pipeline(argc, argv);
  } else {
    g_printerr ("Invalid argument %s. Exiting.\n", argv[1]);
    g_printerr ("Usage: %s <server/s> <url> <domain_socket_path>\n", argv[0]);
    g_printerr ("OR: %s <client/c> <domain_socket_path>\n", argv[0]);
    return -1;
  }

  return ret;
}

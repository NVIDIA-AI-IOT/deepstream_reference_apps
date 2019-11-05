/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <gmodule.h>
#include <string.h>
#include <sys/time.h>
#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"
#include "nvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2
#define SET_GPU_ID(object, gpu_id) g_object_set (G_OBJECT (object), "gpu-id", gpu_id, NULL);
#define SET_MEMORY(object, mem_id) g_object_set (G_OBJECT (object), "nvbuf-memory-type", mem_id, NULL);
#define SINK_ELEMENT "nveglglessink"

GMainLoop *loop = NULL;
/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720
#define GPU_ID 0
#define MAX_NUM_SOURCES 4
#define PGIE_CONFIG_FILE "dstest_pgie_config.txt"
#define TRACKER_CONFIG_FILE "dstest_tracker_config.txt"
#define SGIE1_CONFIG_FILE "dstest_sgie1_config.txt"
#define SGIE2_CONFIG_FILE "dstest_sgie2_config.txt"
#define SGIE3_CONFIG_FILE "dstest_sgie3_config.txt"


#define CONFIG_GPU_ID "gpu-id"
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"

gint g_num_sources = 0;
gint g_source_id_list[MAX_NUM_SOURCES];
gboolean g_eos_list[MAX_NUM_SOURCES];
gboolean g_source_enabled[MAX_NUM_SOURCES];
GstElement **g_source_bin_list = NULL;
GMutex eos_lock;

/* Assuming Resnet 10 model packaged in DS SDK */
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
    *sgie1 = NULL, *sgie2 = NULL, *sgie3 = NULL,
    *nvvideoconvert = NULL, *nvosd = NULL, *tiler = NULL, *tracker = NULL;

gchar *uri = NULL;

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("decodebin child added %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strrstr (name, "nvv4l2decoder") == name) {
#ifdef PLATFORM_TEGRA
    g_object_set (object, "enable-max-performance", TRUE, NULL);
    g_object_set (object, "bufapi-version", TRUE, NULL);
    g_object_set (object, "drop-frame-interval", 0, NULL);
    g_object_set (object, "num-extra-surfaces", 0, NULL);
#else
    g_object_set (object, "gpu-id", GPU_ID, NULL);
#endif
  }
}

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  /* Return absolute path of config file if file_path is NULL. */
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}


static void
cb_newpad (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  g_print ("decodebin new pad %s\n", name);
  if (!strncmp (name, "video", 5)) {
    gint source_id = (*(gint *) data);
    gchar pad_name[16] = { 0 };
    GstPad *sinkpad = NULL;
    g_snprintf (pad_name, 15, "sink_%u", source_id);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
      g_print ("Failed to link decodebin to pipeline\n");
    } else {
      g_print ("Decodebin linked to pipeline\n");
    }
    gst_object_unref (sinkpad);
  }
}

static GstElement *
create_uridecode_bin (guint index, gchar * filename)
{
  GstElement *bin = NULL;
  gchar bin_name[16] = { };

  g_print ("creating uridecodebin for [%s]\n", filename);
  g_source_id_list[index] = index;
  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  bin = gst_element_factory_make ("uridecodebin", bin_name);
  g_object_set (G_OBJECT (bin), "uri", filename, NULL);
  g_signal_connect (G_OBJECT (bin), "pad-added",
      G_CALLBACK (cb_newpad), &g_source_id_list[index]);
  g_signal_connect (G_OBJECT (bin), "child-added",
      G_CALLBACK (decodebin_child_added), &g_source_id_list[index]);
  g_source_enabled[index] = TRUE;

  return bin;
}

static void
stop_release_source (gint source_id)
{
  GstStateChangeReturn state_return;
  gchar pad_name[16];
  GstPad *sinkpad = NULL;
  state_return =
      gst_element_set_state (g_source_bin_list[source_id], GST_STATE_NULL);
  switch (state_return) {
    case GST_STATE_CHANGE_SUCCESS:
      g_print ("STATE CHANGE SUCCESS\n\n");
      g_snprintf (pad_name, 15, "sink_%u", source_id);
      sinkpad = gst_element_get_static_pad (streammux, pad_name);
      gst_pad_send_event (sinkpad, gst_event_new_flush_stop (FALSE));
      gst_element_release_request_pad (streammux, sinkpad);
      g_print ("STATE CHANGE SUCCESS %p\n\n", sinkpad);
      gst_object_unref (sinkpad);
      gst_bin_remove (GST_BIN (pipeline), g_source_bin_list[source_id]);
      source_id--;
      g_num_sources--;
      break;
    case GST_STATE_CHANGE_FAILURE:
      g_print ("STATE CHANGE FAILURE\n\n");
      break;
    case GST_STATE_CHANGE_ASYNC:
      g_print ("STATE CHANGE ASYNC\n\n");
      state_return =
          gst_element_get_state (g_source_bin_list[source_id], NULL, NULL,
          GST_CLOCK_TIME_NONE);
      g_snprintf (pad_name, 15, "sink_%u", source_id);
      sinkpad = gst_element_get_static_pad (streammux, pad_name);
      gst_pad_send_event (sinkpad, gst_event_new_flush_stop (FALSE));
      gst_element_release_request_pad (streammux, sinkpad);
      g_print ("STATE CHANGE ASYNC %p\n\n", sinkpad);
      gst_object_unref (sinkpad);
      gst_bin_remove (GST_BIN (pipeline), g_source_bin_list[source_id]);
      gst_object_unref (g_source_bin_list[source_id]);
      source_id--;
      g_num_sources--;
      break;
    case GST_STATE_CHANGE_NO_PREROLL:
      g_print ("STATE CHANGE NO PREROLL\n\n");
      break;
    default:
      break;
  }


}

static gboolean
delete_sources (gpointer data)
{
  gint source_id;
  g_mutex_lock (&eos_lock);
  for (source_id = 0; source_id < MAX_NUM_SOURCES; source_id++) {
    if (g_eos_list[source_id] == TRUE && g_source_enabled[source_id] == TRUE) {
      g_source_enabled[source_id] = FALSE;
      stop_release_source (source_id);
    }
  }
  g_mutex_unlock (&eos_lock);

  if (g_num_sources == 0) {
    g_main_loop_quit (loop);
    g_print ("All sources Stopped quitting\n");
    return FALSE;
  }

  do {
    source_id = rand () % MAX_NUM_SOURCES;
  } while (!g_source_enabled[source_id]);
  g_source_enabled[source_id] = FALSE;
  g_print ("Calling Stop %d \n", source_id);
  stop_release_source (source_id);

  if (g_num_sources == 0) {
    g_main_loop_quit (loop);
    g_print ("All sources Stopped quitting\n");
    return FALSE;
  }

  return TRUE;
}

static gboolean
add_sources (gpointer data)
{
  gint source_id = g_num_sources;
  GstElement *source_bin;
  GstStateChangeReturn state_return;

  do {
    /* Generating random source id between 0 - MAX_NUM_SOURCES - 1,
     * which has not been enabled
     */
    source_id = rand () % MAX_NUM_SOURCES;
  } while (g_source_enabled[source_id]);
  g_source_enabled[source_id] = TRUE;

  g_print ("Calling Start %d \n", source_id);
  source_bin = create_uridecode_bin (source_id, uri);
  if (!source_bin) {
    g_printerr ("Failed to create source bin. Exiting.\n");
    return -1;
  }
  g_source_bin_list[source_id] = source_bin;
  gst_bin_add (GST_BIN (pipeline), source_bin);
  state_return =
      gst_element_set_state (g_source_bin_list[source_id], GST_STATE_PLAYING);
  switch (state_return) {
    case GST_STATE_CHANGE_SUCCESS:
      g_print ("STATE CHANGE SUCCESS\n\n");
      source_id++;
      break;
    case GST_STATE_CHANGE_FAILURE:
      g_print ("STATE CHANGE FAILURE\n\n");
      break;
    case GST_STATE_CHANGE_ASYNC:
      g_print ("STATE CHANGE ASYNC\n\n");
      state_return =
          gst_element_get_state (g_source_bin_list[source_id], NULL, NULL,
          GST_CLOCK_TIME_NONE);
      source_id++;
      break;
    case GST_STATE_CHANGE_NO_PREROLL:
      g_print ("STATE CHANGE NO PREROLL\n\n");
      break;
    default:
      break;
  }
  g_num_sources++;


  if (g_num_sources == MAX_NUM_SOURCES) {
    /* We have reached MAX_NUM_SOURCES to be added, no stop calling this function
     * and enable calling delete sources
     */
    g_timeout_add_seconds (10, delete_sources, (gpointer) g_source_bin_list);
    return FALSE;
  }

  return TRUE;
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
          g_mutex_lock (&eos_lock);
          g_eos_list[stream_id] = TRUE;
          g_mutex_unlock (&eos_lock);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}


/* Tracker config parsing */

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      guint gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

int
main (int argc, char *argv[])
{
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
#ifdef PLATFORM_TEGRA
  GstElement *nvtransform;
#endif

  /* Check input arguments */
  if (argc != 2) {
    g_printerr ("Usage: %s <uri1> \n", argv[0]);
    return -1;
  }
  num_sources = argc - 1;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  g_mutex_init (&eos_lock);
  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest-pipeline");

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  g_object_set (G_OBJECT (streammux), "batched-push-timeout", 25000, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", 30, NULL);
  SET_GPU_ID (streammux, GPU_ID);

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);
  g_object_set (G_OBJECT (streammux), "live-source", 1, NULL);

  g_source_bin_list = g_malloc0 (sizeof (GstElement *) * MAX_NUM_SOURCES);
  uri = g_strdup (argv[1]);
  for (i = 0; i < /*num_sources */ 1; i++) {
    GstElement *source_bin = create_uridecode_bin (i, argv[i + 1]);
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }
    g_source_bin_list[i] = source_bin;
    gst_bin_add (GST_BIN (pipeline), source_bin);
  }

  g_num_sources = num_sources;

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file
   */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use nvtiler to stitch o/p from upstream components */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvideoconvert =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
#ifdef PLATFORM_TEGRA
  nvtransform = gst_element_factory_make ("nvegltransform", "nvegltransform");
#endif

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  tracker = gst_element_factory_make ("nvtracker", "nvtracker");

  sgie1 = gst_element_factory_make ("nvinfer", "secondary-nvinference-engine1");
  sgie2 = gst_element_factory_make ("nvinfer", "secondary-nvinference-engine2");
  sgie3 = gst_element_factory_make ("nvinfer", "secondary-nvinference-engine3");

  /* Finally render the osd output */
  sink = gst_element_factory_make (SINK_ELEMENT, "nveglglessink");

  if (!pgie || !sgie1 || !sgie2 || !sgie3 || !tiler || !nvvideoconvert || !nvosd
      || !sink || !tracker
#ifdef PLATFORM_TEGRA
      || !nvtransform
#endif
      ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie1), "config-file-path", SGIE1_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie2), "config-file-path", SGIE2_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie3), "config-file-path", SGIE3_CONFIG_FILE, NULL);

  /* Set necessary properties of the tracker element. */
  if (!set_tracker_properties(tracker)) {
    g_printerr ("Failed to set tracker properties. Exiting.\n");
    return -1;
  }

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size < MAX_NUM_SOURCES) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", MAX_NUM_SOURCES, NULL);
  }

  /* Set GPU ID of elements */
  SET_GPU_ID (pgie, GPU_ID);
  SET_GPU_ID (sgie1, GPU_ID);
  SET_GPU_ID (sgie2, GPU_ID);
  SET_GPU_ID (sgie3, GPU_ID);

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the osd properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);
  SET_GPU_ID (tiler, GPU_ID);
  SET_GPU_ID (nvvideoconvert, GPU_ID);
  SET_GPU_ID (nvosd, GPU_ID);
#ifndef PLATFORM_TEGRA
  SET_GPU_ID (sink, GPU_ID);
#endif

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), pgie, tracker, sgie1, sgie2, sgie3,
      tiler, nvvideoconvert, nvosd, sink, NULL);

#ifdef PLATFORM_TEGRA
  gst_bin_add (GST_BIN (pipeline), nvtransform);
#endif

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvideoconvert -> nvosd -> video-renderer */
#ifdef PLATFORM_TEGRA
  if (!gst_element_link_many (streammux, pgie, tracker, sgie1, sgie2, sgie3,
          tiler, nvvideoconvert, nvosd, nvtransform, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
  if (!gst_element_link_many (streammux, pgie, tracker, sgie1, sgie2, sgie3,
          tiler, nvvideoconvert, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif

  g_object_set (G_OBJECT (sink), "sync", FALSE, "qos", FALSE, NULL);

  gst_element_set_state (pipeline, GST_STATE_PAUSED);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[1]);
  if (gst_element_set_state (pipeline,
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    g_printerr ("Failed to set pipeline to playing. Exiting.\n");
    return -1;
  }
  //GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-playing");

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_timeout_add_seconds (10, add_sources, (gpointer) g_source_bin_list);
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  g_free (g_source_bin_list);
  g_free (uri);
  g_mutex_clear (&eos_lock);
  return 0;
}

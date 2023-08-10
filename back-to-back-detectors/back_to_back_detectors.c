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
#include "gstnvdsmeta.h"
#include <cuda_runtime_api.h>

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define SGIE_CLASS_ID_LP 1
#define SGIE_CLASS_ID_FACE 0

/* Change this to 0 to make the 2nd detector act as a primary(full-frame) detector.
 * When set to 1, it will act as secondary(operates on primary detected objects). */
#define SECOND_DETECTOR_IS_SECONDARY 1

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

#define INFER_PGIE_CONFIG_FILE "primary_detector_config.txt"
#define INFER_SGIE_CONFIG_FILE "secondary_detector_config.txt"
#define INFERSERVER_PGIE_CONFIG_FILE "inferserver/primary_detector_config.txt"
#define INFERSERVER_SGIE_CONFIG_FILE "inferserver/secondary_detector_config.txt"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

#define PRIMARY_DETECTOR_UID 1
#define SECONDARY_DETECTOR_UID 2

/* nvvidconv_sink_pad_buffer_probe  will extract metadata received on nvvideoconvert sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
nvvidconv_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint face_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);

            /* Check that the object has been detected by the primary detector
             * and that the class id is that of vehicles/persons. */
            if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
              if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
                vehicle_count++;
              if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
                person_count++;
            }

            if (obj_meta->unique_component_id == SECONDARY_DETECTOR_UID) {
              if (obj_meta->class_id == SGIE_CLASS_ID_FACE) {
                face_count++;
                /* Print this info only when operating in secondary model. */
                if (SECOND_DETECTOR_IS_SECONDARY)
                  g_print ("Face found for parent object %p (type=%s)\n",
                      obj_meta->parent, pgie_classes_str[obj_meta->parent->class_id]);
              }
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset += snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);
        offset += snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Face = %d ", face_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }


    g_print ("Frame Number = %d Vehicle Count = %d Person Count = %d"
            " Face Count = %d\n",
            frame_number, vehicle_count, person_count,
            face_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
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
    case GST_MESSAGE_ERROR:{
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
    default:
      break;
  }
  return TRUE;
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s <h264_elementary_stream>\n", bin);
  g_printerr ("For nvinferserver, Usage: %s -t inferserver <h264_elementary_stream>\n", bin);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *streammux = NULL, *sink = NULL, *primary_detector = NULL,
      *secondary_detector = NULL, *nvvidconv = NULL, *nvosd = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *nvvidconv_sink_pad = NULL;
  gboolean is_nvinfer_server = FALSE;
  gchar *input_stream = NULL;
  const char *infer_plugin = NVINFER_PLUGIN;

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
    infer_plugin = NVINFERSERVER_PLUGIN;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Create two nvinfer instances for the two back-to-back detectors */
  primary_detector = gst_element_factory_make (infer_plugin, "primary-nvinference-engine1");

  secondary_detector = gst_element_factory_make (infer_plugin, "primary-nvinference-engine2");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if(prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
  } else {
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  }

  if (!source || !h264parser || !decoder || !primary_detector || !secondary_detector
      || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  if (is_nvinfer_server) {
    input_stream = argv[3];
    g_object_set (G_OBJECT (source), "location", argv[3], NULL);
  } else {
    input_stream = argv[1];
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);
  }

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", 1,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set the config files for the two detectors. We demonstrate this by using
   * the same detector model twice but making them act as vehicle-only and
   * person-only detectors by adjusting the bbox confidence thresholds in the
   * two seperate config files. */
  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (primary_detector), "config-file-path", INFERSERVER_PGIE_CONFIG_FILE,
      "unique-id", PRIMARY_DETECTOR_UID, NULL);
  } else {
    g_object_set (G_OBJECT (primary_detector), "config-file-path", INFER_PGIE_CONFIG_FILE,
      "unique-id", PRIMARY_DETECTOR_UID, NULL);
  }

  if (is_nvinfer_server) {
    g_object_set (G_OBJECT (secondary_detector), "config-file-path", INFERSERVER_SGIE_CONFIG_FILE,
      "unique-id", SECONDARY_DETECTOR_UID, "process-mode", SECOND_DETECTOR_IS_SECONDARY ? 2 : 1, NULL);
  } else {
    g_object_set (G_OBJECT (secondary_detector), "config-file-path", INFER_SGIE_CONFIG_FILE,
      "unique-id", SECONDARY_DETECTOR_UID, "process-mode", SECOND_DETECTOR_IS_SECONDARY ? 2 : 1, NULL);
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, primary_detector, secondary_detector,
      nvvidconv, nvosd, sink, NULL);

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (decoder, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * pgie -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (streammux, primary_detector, secondary_detector,
      nvvidconv, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the nvvideoconvert element, since by that time, the buffer would have
   * had got all the metadata. */
  nvvidconv_sink_pad = gst_element_get_static_pad (nvvidconv, "sink");
  if (!nvvidconv_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (nvvidconv_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        nvvidconv_sink_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", input_stream);
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
  return 0;
}

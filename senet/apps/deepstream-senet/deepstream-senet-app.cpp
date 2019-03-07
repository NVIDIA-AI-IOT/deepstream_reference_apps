/*
 * Copyright (c) 2018 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <gst/gst.h>
#include <glib.h>

#include <iostream>
#include <cassert>
#include <experimental/filesystem>

#include "gstnvdsmeta.h"

#define PGIE_CONFIG_FILE  "config/config_infer_primary_resnet10.txt"
#define PGIE_LABELS "data/resnet10_labels.txt"
#define SGIE1_LABELS "data/imagenet_labels.txt"
#define MAX_DISPLAY_LEN 128

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* gie_unique_id is one of the properties in the above config_infer_secondary_senet.txt and config_infer_primary_resnet10.txt
 * files. These should be unique and known when we want to parse the Metadata
 * respective to the sgie labels. Ideally these should be read from the config
 * files but for brevity we ensure they are same. */
guint pgie_unique_id = 1;
guint sgie1_unique_id = 2;

/* These are the strings of the labels for the respective models */
static std::vector<std::string> sgie1_classes_str;
static std::vector<std::string> pgie_classes_str;

bool
fileExists (const std::string fileName)
{
  if (!std::experimental::filesystem::exists (std::experimental::
          filesystem::path (fileName))) {
    std::cout << "File does not exist : " << fileName << std::endl;
    return false;
  }
  return true;
}

std::vector<std::string> loadListFromTextFile (const std::string filename)
{
  assert (fileExists (filename));
  std::vector < std::string > list;

  FILE *f = fopen (filename.c_str (), "r");
  if (!f) {
    std::cout << "failed to open " << filename;
    assert (0);
  }

  char str[512];
  while (fgets (str, 512, f) != NULL) {
    for (int i = 0; str[i] != '\0'; ++i) {
      if (str[i] == '\n') {
        str[i] = '\0';
        break;
      }
    }
    list.push_back (str);
  }
  fclose (f);
  return list;
}

/* This is the buffer probe function that we have registered on the sink pad
 * of the OSD element. All the infer elements in the pipeline shall attach
 * their metadata to the GstBuffer, here we will iterate & process the metadata
 * forex: class ids to strings, counting of class_id objects etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstMeta *gst_meta = NULL;
  NvDsMeta *nvdsmeta = NULL;
  gpointer state = NULL;
  static GQuark _nvdsmeta_quark = 0;
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsFrameMeta *frame_meta = NULL;
  guint num_rects = 0, rect_index = 0;
  NvDsObjectParams *obj_meta = NULL;
  gint sgie1_class_id = -1;
  NvDsAttrInfo *sgie1_attrs = NULL;
  guint tracking_id = 0;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvOSD_TextParams *txt_params = NULL;

  static guint frame_number = 0;

  if (!_nvdsmeta_quark)
    _nvdsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

  while ((gst_meta = gst_buffer_iterate_meta (buf, &state))) {
    if (gst_meta_api_type_has_tag (gst_meta->info->api, _nvdsmeta_quark)) {

      nvdsmeta = (NvDsMeta *) gst_meta;

      /* We are interested only in intercepting Meta of type
       * "NVDS_META_FRAME_INFO" as they are from our infer elements. */
      if (nvdsmeta->meta_type == NVDS_META_FRAME_INFO) {
        frame_meta = (NvDsFrameMeta *) nvdsmeta->meta_data;
        if (frame_meta == NULL) {
          g_print ("NVDS Meta contained NULL meta \n");
          return GST_PAD_PROBE_OK;
        }

        /* We reset the num_strings here as we plan to iterate through the
         *  the detected objects and form our own strings by mapping the sgie
         *  class id label & its label string. The pipeline generated strings
         *  shall be discarded.
         */
        frame_meta->num_strings = 0;

        num_rects = frame_meta->num_rects;
        /* This means we have num_rects in frame_meta->obj_params,
         * now lets iterate through them */

        for (rect_index = 0; rect_index < num_rects; rect_index++) {

          /* Reset class string indexes per object detected */
          sgie1_class_id = -1;

          obj_meta = (NvDsObjectParams *) & frame_meta->obj_params[rect_index];

          if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
            vehicle_count++;
          if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
            person_count++;

          /* Now for each of the obj_meta, we iterate through the attr_infos.
           * gie_unique_id of the sgie instance should be known and it is
           * the index into the obj_meta->attr_info */
          sgie1_attrs = &(obj_meta->attr_info[sgie1_unique_id]);
          if (!g_strcmp0 (obj_meta->attr_info[sgie1_unique_id].attr_label,
                  "car"))
            vehicle_count++;
          else if (!g_strcmp0 (obj_meta->attr_info[sgie1_unique_id].attr_label,
                  "person"))
            person_count++;

          /* Our classifiers are single-class classifiers, hence num_attrs
           * should always be 1, for multi-class secondary classifiers, we
           * would have to iterate through the attr_info to get the respective
           * strings. */
          if (sgie1_attrs->num_attrs)
            sgie1_class_id = sgie1_attrs->attrs[0].attr_val;

          tracking_id = obj_meta->tracking_id;

          /* Now using above information we need to form a text that should
           * be displayed on top of the bounding box, so lets form it here.
           * We shall free any previously generated text by the pipeline, hence
           * free it first. */
          txt_params = &(obj_meta->text_params);
          if (txt_params->display_text)
            g_free (txt_params->display_text);

          txt_params->display_text = (char*) g_malloc0 (MAX_DISPLAY_LEN);

          g_snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "%s ",
              pgie_classes_str[obj_meta->class_id].c_str());

          if (sgie1_class_id != -1) {
            g_strlcat (txt_params->display_text, ": ", MAX_DISPLAY_LEN);
            g_strlcat (txt_params->display_text,
                sgie1_classes_str[sgie1_class_id].c_str(), MAX_DISPLAY_LEN);
          }

          /* Now set the offsets where the string should appear */
          txt_params->x_offset = obj_meta->rect_params.left;
          txt_params->y_offset = obj_meta->rect_params.top - 25;

          /* Font , font-color and font-size */
          txt_params->font_params.font_name = "Arial";
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

          frame_meta->num_strings++;
        }
        g_print ("Frame Number = %d Number of objects = %d"
            " Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
      }
    }
  }
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
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n", GST_OBJECT_NAME (msg->src),
          error->message);
      g_free (debug);
      g_printerr ("Error: %s\n", error->message);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *decoder =
      NULL, *sink = NULL, *nvvidconv = NULL, *nvosd = NULL, *pgie = NULL, *sgie = NULL, *nvtracker = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL;
  gulong osd_probe_id = 0;

  /* Check input arguments */
  if (argc != 4) {
    g_printerr
        ("Usage: %s <Platform-Telsa/Tegra> <H264 filename> <senet-plugin config file> \n",
        argv[0]);
    return -1;
  }

  pgie_classes_str = loadListFromTextFile (PGIE_LABELS);
  sgie1_classes_str = loadListFromTextFile (SGIE1_LABELS);

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */

  /* Create Pipeline element that will be a container of other elements */
  pipeline = gst_pipeline_new ("ds-resnet-senet-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264/omxh264dec for hardware accelerated decode on GPU */
  if (!g_strcmp0 ("Tesla", argv[1])) {
    decoder = gst_element_factory_make ("nvdec_h264", "nvh264-decoder");
  } else if (!g_strcmp0 ("Tegra", argv[1])) {
    decoder = gst_element_factory_make ("omxh264dec", "openmax-decoder");
  } else {
    g_printerr ("Incorrect platform. Choose between Telsa/Tegra. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* We need to have a tracker to track the identified objects */
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");

  /* We need three secondary gies so lets create one more instances of nvinfer */
  sgie =  gst_element_factory_make ("nvinfer", "secondary-senet-inference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvidconv", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if (!g_strcmp0 ("Tesla", argv[1])) {
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  } else if (!g_strcmp0 ("Tegra", argv[1])) {
    sink = gst_element_factory_make ("nvoverlaysink", "nvvideo-renderer");
  } else {
    g_printerr ("Incorrect platform. Choose between Telsa/Tegra. Exiting.\n");
    return -1;
  }
  /* caps filter for nvvidconv to convert NV12 to RGBA as nvosd expects input
   * in RGBA format */
  if (!pipeline || !source || !h264parser || !decoder || !pgie || !nvtracker ||
      !sgie || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  g_object_set (G_OBJECT (source), "location", argv[2], NULL);
  g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie), "config-file-path", argv[3], NULL);

  /* we set the osd properties here */
  g_object_set (G_OBJECT (nvosd), "font-size", 15, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* decoder | pgie1 | nvtracker | sgie1 | sgie2 | sgie3 | etc.. */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, pgie, nvtracker, sgie,
      nvvidconv, nvosd, sink, NULL);

  /* Link the elements together */
  if (!gst_element_link_many (source, h264parser, decoder, pgie, nvtracker, sgie,
       nvvidconv, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    osd_probe_id = gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[2]);
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

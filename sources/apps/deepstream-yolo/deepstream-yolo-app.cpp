/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"

/* As defined in the yolo plugins header*/

#define YOLO_UNIQUE_ID 15

gint frame_number = 0;

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and get a count of objects of interest */

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
  guint car_count = 0;
  guint person_count = 0;
  guint bicycle_count = 0;
  guint truck_count = 0;

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
          g_print ("NvDS Meta contained NULL meta \n");
          return GST_PAD_PROBE_OK;
        }

        num_rects = frame_meta->num_rects;

        /* This means we have num_rects in frame_meta->obj_params.
         * Now lets iterate through them and count the number of cars,
         * trucks, persons and bicycles in each frame */

        for (rect_index = 0; rect_index < num_rects; rect_index++) {
          obj_meta = (NvDsObjectParams *) & frame_meta->obj_params[rect_index];
          if (!g_strcmp0 (obj_meta->attr_info[YOLO_UNIQUE_ID].attr_label,
                  "car"))
            car_count++;
          else if (!g_strcmp0 (obj_meta->attr_info[YOLO_UNIQUE_ID].attr_label,
                  "person"))
            person_count++;
          else if (!g_strcmp0 (obj_meta->attr_info[YOLO_UNIQUE_ID].attr_label,
                  "bicycle"))
            bicycle_count++;
          else if (!g_strcmp0 (obj_meta->attr_info[YOLO_UNIQUE_ID].attr_label,
                  "truck"))
            truck_count++;
        }
      }
    }
  }
  g_print ("Frame Number = %d Number of objects = %d "
      "Car Count = %d Person Count = %d "
      "Bicycle Count = %d Truck Count = %d \n",
      frame_number, num_rects, car_count, person_count, bicycle_count,
      truck_count);
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
      NULL, *sink = NULL, *nvvidconv = NULL, *nvosd = NULL, *filter1 =
      NULL, *filter2 = NULL, *yolo = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstCaps *caps1 = NULL, *caps2 = NULL;
  gulong osd_probe_id = 0;
  GstPad *osd_sink_pad = NULL;

  /* Check input arguments */
  if (argc != 4) {
    g_printerr
        ("Usage: %s <Platform-Telsa/Tegra> <H264 filename> <yolo-plugin config file> \n",
        argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-yolo-pipeline");

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

  /* Use convertor to convert from NV12 to RGBA as required by nvosd and yolo plugins */
  nvvidconv = gst_element_factory_make ("nvvidconv", "nvvideo-converter");

  /* Use yolo to run inference instead of pgie */
  yolo = gst_element_factory_make ("nvyolo", "yolo-inference-engine");

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
  filter1 = gst_element_factory_make ("capsfilter", "filter1");
  filter2 = gst_element_factory_make ("capsfilter", "filter2");
  if (!pipeline || !source || !h264parser || !decoder || !filter1 || !nvvidconv
      || !filter2 || !nvosd || !sink || !yolo) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  g_object_set (G_OBJECT (source), "location", argv[2], NULL);
  g_object_set (G_OBJECT (yolo), "config-file-path", argv[3], NULL);

  /* we set the osd properties here */
  g_object_set (G_OBJECT (nvosd), "font-size", 15, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), source, h264parser, decoder, filter1,
      nvvidconv, filter2, yolo, nvosd, sink, NULL);
  caps1 = gst_caps_from_string ("video/x-raw(memory:NVMM), format=NV12");
  g_object_set (G_OBJECT (filter1), "caps", caps1, NULL);
  gst_caps_unref (caps1);
  caps2 = gst_caps_from_string ("video/x-raw(memory:NVMM), format=RGBA");
  g_object_set (G_OBJECT (filter2), "caps", caps2, NULL);
  gst_caps_unref (caps2);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * filter1 -> nvvidconv -> filter2 -> yolo -> nvosd -> video-renderer */
  gst_element_link_many (source, h264parser, decoder, filter1, nvvidconv,
      filter2, yolo, nvosd, sink, NULL);

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

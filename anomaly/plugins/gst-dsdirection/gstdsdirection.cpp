/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
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

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/time.h>
#include "gstdsdirection.h"
#include "nvds_opticalflow_meta.h"
GST_DEBUG_CATEGORY_STATIC (gst_dsdirection_debug);
#define GST_CAT_DEFAULT gst_dsdirection_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID
};
// Block size used in NVOF application
#define NVOF_BLK_SIZE 4
/*set the user metadata type*/
#define NVDS_DIRECTION_USER_META (nvds_get_user_meta_type(((gchar *)"NVIDIA.NVDSDIRECTION.DIR_META")))
/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsdirection_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsdirection_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsdirection_parent_class parent_class
G_DEFINE_TYPE (GstDsDirection, gst_dsdirection, GST_TYPE_BASE_TRANSFORM);

static void gst_dsdirection_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsdirection_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_dsdirection_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);


static void attach_metadata_object (GstDsDirection * dsdirection,
    NvDsObjectMeta * obj_meta, DsDirectionOutput * output);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsdirection_class_init (GstDsDirectionClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_dsdirection_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_dsdirection_get_property);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dsdirection_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsdirection_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsdirection_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsDirection plugin",
      "DsDirection Plugin",
      "Estimate direction in which object is moving",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dsdirection_init (GstDsDirection * dsdirection)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsdirection);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dsdirection->unique_id = DEFAULT_UNIQUE_ID;
  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
*/
static void
gst_dsdirection_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsDirection *dsdirection = GST_DSDIRECTION (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dsdirection->unique_id = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dsdirection_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsDirection *dsdirection = GST_DSDIRECTION (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dsdirection->unique_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsdirection_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsDirection *dsdirection = GST_DSDIRECTION (btrans);
  DsDirectionOutput *output;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsObjectMeta *obj_meta = NULL;
  dsdirection->frame_num++;

  GST_DEBUG_OBJECT (dsdirection,
      "Processing Frame %lu", dsdirection->frame_num);

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsdirection, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  //Iterating through frames in batched meta from diff. sources
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {

    frame_meta = (NvDsFrameMeta *) (l_frame->data);
    NvDsFrameMetaList *fmeta_list = NULL;
    //Iterating through each frame of the batched meta
    for (fmeta_list = frame_meta->frame_user_meta_list; fmeta_list != NULL;
        fmeta_list = fmeta_list->next) {
      NvDsUserMeta *of_user_meta = NULL;
      //previous meta + optical flow meta
      of_user_meta = (NvDsUserMeta *) fmeta_list->data;

      if (of_user_meta
          && of_user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META) {
        //optical flow meta for each frame
        NvDsOpticalFlowMeta *ofmeta =
            (NvDsOpticalFlowMeta *) (of_user_meta->user_meta_data);
        if (ofmeta) {
          //Iterating through each object in the frame
          for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
              l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);

            //processing the meta data for direction detection
            output =
                DsDirectionProcess ((NvOFFlowVector *) ofmeta->data,
                ofmeta->cols, ofmeta->rows, NVOF_BLK_SIZE,
                &obj_meta->rect_params);
            // Attach direction to the object
            attach_metadata_object (dsdirection, obj_meta, output);

          }
        }
      }
    }
  }
  return GST_FLOW_OK;
}

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer
copy_ds_direction_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  DsDirectionOutput *src_user_metadata
      = (DsDirectionOutput *) user_meta->user_meta_data;
  DsDirectionOutput *dst_user_metadata
      = (DsDirectionOutput *) calloc (1, sizeof (DsDirectionOutput));
  memcpy (dst_user_metadata, src_user_metadata, sizeof (DsDirectionOutput));
  return (gpointer) dst_user_metadata;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void
release_ds_direction_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if (user_meta->user_meta_data) {
    free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void
attach_metadata_object (GstDsDirection * dsdirection, NvDsObjectMeta * obj_meta,
    DsDirectionOutput * output)
{
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

  // Attach - DsDirection MetaData
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
  NvDsMetaType user_meta_type = NVDS_DIRECTION_USER_META;

  user_meta->user_meta_data = output;
  user_meta->base_meta.meta_type = user_meta_type;
  user_meta->base_meta.copy_func = copy_ds_direction_meta;
  user_meta->base_meta.release_func = release_ds_direction_meta;

  nvds_add_user_meta_to_obj (obj_meta, user_meta);

  nvds_acquire_meta_lock (batch_meta);
  NvOSD_TextParams & text_params = obj_meta->text_params;
  NvOSD_RectParams & rect_params = obj_meta->rect_params;

  /* Below code to display the result */
  // Set black background for the text
  // display_text required heap allocated memory
  if (text_params.display_text) {
    gchar *conc_string = g_strconcat (text_params.display_text, " ",
        output->object.direction, NULL);
    g_free (text_params.display_text);
    text_params.display_text = conc_string;
    text_params.font_params.font_size = 12;
  } else {
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    text_params.display_text = g_strdup (output->object.direction);
    // Font face, size and color
    text_params.font_params.font_name = (char *) "Serif";
    text_params.font_params.font_size = 15;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
  }
  nvds_release_meta_lock (batch_meta);
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsdirection_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsdirection_debug, "dsdirection", 0,
      "dsdirection plugin");

  return gst_element_register (plugin, "dsdirection", GST_RANK_PRIMARY,
      GST_TYPE_DSDIRECTION);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsdirection,
    DESCRIPTION, dsdirection_plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE,
    URL)

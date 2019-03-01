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

#include "gstyoloplugin.h"
#include <fstream>
#include <iostream>
#include <npp.h>
#include <ostream>
#include <sstream>
#include <string.h>
#include <string>
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC (gst_yoloplugin_debug);
#define GST_CAT_DEFAULT gst_yoloplugin_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_GPU_DEVICE_ID,
  PROP_CONFIG_FILE_PATH
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_GPU_ID 0
#define DEFAULT_CONFIG_FILE_PATH ""

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_yoloplugin_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ RGBA }")));

static GstStaticPadTemplate gst_yoloplugin_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_yoloplugin_parent_class parent_class
G_DEFINE_TYPE (GstYoloPlugin, gst_yoloplugin, GST_TYPE_BASE_TRANSFORM);

static void gst_yoloplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_yoloplugin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_yoloplugin_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_yoloplugin_start (GstBaseTransform * btrans);
static gboolean gst_yoloplugin_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_yoloplugin_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf);

static void attach_metadata_full_frame (GstYoloPlugin * yoloplugin,
    GstBuffer * inbuf, gdouble scale_ratio, YoloPluginOutput * output,
    guint batch_id);
static void attach_metadata_object (GstYoloPlugin * yoloplugin,
    NvDsObjectParams * obj_param, YoloPluginOutput * output);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_yoloplugin_class_init (GstYoloPluginClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_yoloplugin_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_yoloplugin_get_property);

  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_yoloplugin_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_yoloplugin_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_yoloplugin_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_yoloplugin_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element",
          0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width", "Processing Width",
          "Width of the input buffer to algorithm", 1, G_MAXINT,
          DEFAULT_PROCESSING_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height", "Processing Height",
          "Height of the input buffer to algorithm", 1, G_MAXINT,
          DEFAULT_PROCESSING_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame", "Full frame",
          "Enable to process full frame or disable to process objects detected"
          "by primary detector",
          DEFAULT_PROCESS_FULL_FRAME,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Plugin config file path",
          "Set plugin config file path",
          DEFAULT_CONFIG_FILE_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_yoloplugin_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_yoloplugin_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvYolo", "NvYolo",
      "Process a 3rdparty example algorithm on objects / full frame", "Nvidia");
}

static void
gst_yoloplugin_init (GstYoloPlugin * yoloplugin)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (yoloplugin);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  yoloplugin->unique_id = DEFAULT_UNIQUE_ID;
  yoloplugin->processing_width = DEFAULT_PROCESSING_WIDTH;
  yoloplugin->processing_height = DEFAULT_PROCESSING_HEIGHT;
  yoloplugin->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  yoloplugin->gpu_id = DEFAULT_GPU_ID;
  yoloplugin->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);
  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_yoloplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      yoloplugin->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      yoloplugin->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      yoloplugin->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      yoloplugin->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      yoloplugin->gpu_id = g_value_get_uint (value);
      break;
    case PROP_CONFIG_FILE_PATH:
      if (g_value_get_string (value)) {
        g_free (yoloplugin->config_file_path);
        yoloplugin->config_file_path = g_value_dup_string (value);
      }
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
gst_yoloplugin_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, yoloplugin->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, yoloplugin->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, yoloplugin->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, yoloplugin->process_full_frame);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, yoloplugin->gpu_id);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, yoloplugin->config_file_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_yoloplugin_start (GstBaseTransform * btrans)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);
  YoloPluginInitParams init_params =
      { yoloplugin->processing_width, yoloplugin->processing_height,
    yoloplugin->process_full_frame, yoloplugin->config_file_path
  };

  GstQuery *queryparams = NULL;
  guint batch_size = 1;

  if ((!yoloplugin->config_file_path)
      || (strlen (yoloplugin->config_file_path) == 0)) {
    g_print ("ERROR: Yolo plugin config file path not set \n");
    goto error;
  }

  yoloplugin->batch_size = 1;
  queryparams = gst_nvquery_batch_size_new ();
  if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), queryparams)
      || gst_pad_peer_query (GST_BASE_TRANSFORM_SRC_PAD (btrans), queryparams)) {
    if (gst_nvquery_batch_size_parse (queryparams, &batch_size)) {
      yoloplugin->batch_size = batch_size;
    }
  }
  GST_DEBUG_OBJECT (yoloplugin, "Setting batch-size %d \n",
      yoloplugin->batch_size);
  gst_query_unref (queryparams);

  /* Algorithm specific initializations and resource allocation. */
  yoloplugin->yolopluginlib_ctx =
      YoloPluginCtxInit (&init_params, yoloplugin->batch_size);

  g_assert (yoloplugin->yolopluginlib_ctx
      && "Unable to create yolo plugin lib ctx \n ");
  GST_DEBUG_OBJECT (yoloplugin, "ctx lib %p \n", yoloplugin->yolopluginlib_ctx);
  CHECK_CUDA_STATUS (cudaSetDevice (yoloplugin->gpu_id),
      "Unable to set cuda device");

  cudaStreamCreate (&yoloplugin->npp_stream);

  // Create host memory for conversion/scaling
  CHECK_CUDA_STATUS (cudaMallocHost (&yoloplugin->hconv_buf,
          yoloplugin->processing_width * yoloplugin->processing_height *
          RGBA_BYTES_PER_PIXEL), "Could not allocate cuda host buffer");

  GST_DEBUG_OBJECT (yoloplugin, "allocated cuda buffer %p \n",
      yoloplugin->hconv_buf);

  yoloplugin->cvmats =
      std::vector < cv::Mat * >(yoloplugin->batch_size, nullptr);
  for (uint k = 0; k < batch_size; ++k) {
    yoloplugin->cvmats.at (k) =
        new cv::Mat (cv::Size (yoloplugin->processing_width,
            yoloplugin->processing_height), CV_8UC3);
    if (!yoloplugin->cvmats.at (k))
      goto error;
  }
  GST_DEBUG_OBJECT (yoloplugin, "created CV Mat\n");
  return TRUE;
error:
  if (yoloplugin->hconv_buf) {
    cudaFreeHost (yoloplugin->hconv_buf);
    yoloplugin->hconv_buf = NULL;
  }

  if (yoloplugin->npp_stream) {
    cudaStreamDestroy (yoloplugin->npp_stream);
    yoloplugin->npp_stream = NULL;
  }
  if (yoloplugin->yolopluginlib_ctx)
    YoloPluginCtxDeinit (yoloplugin->yolopluginlib_ctx);
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_yoloplugin_stop (GstBaseTransform * btrans)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);
  if (yoloplugin->hconv_buf) {
    cudaFreeHost (yoloplugin->hconv_buf);
    yoloplugin->hconv_buf = NULL;
    GST_DEBUG_OBJECT (yoloplugin, "Freed cuda host buffer \n");
  }
  if (yoloplugin->npp_stream) {
    cudaStreamDestroy (yoloplugin->npp_stream);
  }

  for (uint i = 0; i < yoloplugin->batch_size; ++i) {
    delete yoloplugin->cvmats.at (i);
  }
  GST_DEBUG_OBJECT (yoloplugin, "deleted CV Mat \n");
  // Deinit the algorithm library
  YoloPluginCtxDeinit (yoloplugin->yolopluginlib_ctx);
  GST_DEBUG_OBJECT (yoloplugin, "ctx lib released \n");
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_yoloplugin_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&yoloplugin->video_info, incaps);

  CHECK_CUDA_STATUS (cudaSetDevice (yoloplugin->gpu_id),
      "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio. Remove the padding requried by hardware and convert from RGBA to RGB
 * using openCV. These steps can be skipped if the algorithm can work with
 * padded data and/or can work with RGBA.
 */
static GstFlowReturn
get_converted_mat_dgpu (GstYoloPlugin * yoloplugin, void *input_buf,
    NvOSD_RectParams * crop_rect_params, cv::Mat & out_mat,
    gdouble & ratio, gint input_width, gint input_height)
{
  cv::Mat in_mat;
  gint src_left = (crop_rect_params->left);
  gint src_top = (crop_rect_params->top);
  gint src_width = (crop_rect_params->width);
  gint src_height = (crop_rect_params->height);

  // size of source
  NppiSize oSrcSize = { input_width, input_height };

  // source ROI
  NppiRect oSrcROI =
      { (gint) 0, (gint) 0, (gint) src_width, (gint) src_height };

  // Destination ROI
  NppiRect DstROI = { 0, 0, (gint) yoloplugin->processing_width,
    (gint) yoloplugin->processing_height
  };

  GST_DEBUG_OBJECT (yoloplugin, "Scaling and converting input buffer\n");

  // Calculate scaling ratio while maintaining aspect ratio
  ratio = MIN (1.0 * yoloplugin->processing_width / crop_rect_params->width,
      1.0 * yoloplugin->processing_height / crop_rect_params->height);

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    return GST_FLOW_ERROR;
  }
  // Memset the memory
  CHECK_CUDA_STATUS (cudaMemset (yoloplugin->hconv_buf, 0,
          yoloplugin->processing_width * RGBA_BYTES_PER_PIXEL *
          yoloplugin->processing_height), "Failed to memset cuda buffer");

  nppSetStream (yoloplugin->npp_stream);

  // Perform cropping and resizing
  CHECK_NPP_STATUS (nppiResizeSqrPixel_8u_C4R (
          (const Npp8u *) input_buf + (src_left +
              src_top * input_width) * RGBA_BYTES_PER_PIXEL, oSrcSize,
          input_width * RGBA_BYTES_PER_PIXEL, oSrcROI,
          (Npp8u *) yoloplugin->hconv_buf,
          yoloplugin->processing_width * RGBA_BYTES_PER_PIXEL, DstROI, ratio,
          ratio, 0, 0, NPPI_INTER_LINEAR), "Failed to scale RGBA frame");

  CHECK_CUDA_STATUS (cudaStreamSynchronize (yoloplugin->npp_stream),
      "Failed to synchronize cuda stream");

  // Use openCV to remove padding and convert RGBA to RGB. Can be skipped if
  // algorithm can handle padded RGBA data.
  in_mat =
      cv::Mat (yoloplugin->processing_height, yoloplugin->processing_width,
      CV_8UC4, yoloplugin->hconv_buf,
      yoloplugin->processing_width * RGBA_BYTES_PER_PIXEL);
  cv::cvtColor (in_mat, out_mat, CV_RGBA2BGR);

  return GST_FLOW_OK;

error:
  return GST_FLOW_ERROR;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_yoloplugin_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_OK;
  gdouble scale_ratio;
  std::vector < YoloPluginOutput * >outputs (yoloplugin->batch_size, nullptr);

  NvBufSurface *surface = NULL;
  guint batch_size = yoloplugin->batch_size;
  GstNvStreamMeta *streamMeta = NULL;

  cv::Mat in_mat;

  yoloplugin->frame_num++;
  CHECK_CUDA_STATUS (cudaSetDevice (yoloplugin->gpu_id),
      "Unable to set cuda device");

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    goto error;
  }

  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (yoloplugin,
      "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
      yoloplugin->frame_num, surface);

  if (CHECK_NVDS_MEMORY_AND_GPUID (yoloplugin, surface))
    goto error;

  /* Stream meta for batched mode */
  streamMeta = gst_buffer_get_nvstream_meta (inbuf);
  if (streamMeta) {
    batch_size = MIN (streamMeta->num_filled, batch_size);
  }
  if (yoloplugin->process_full_frame) {
    for (guint i = 0; i < batch_size; i++) {
      NvOSD_RectParams rect_params;

      // Scale the entire frame to processing resolution
      rect_params.left = 0;
      rect_params.top = 0;
      rect_params.width = yoloplugin->video_info.width;
      rect_params.height = yoloplugin->video_info.height;

      // Scale and convert the frame
      if (get_converted_mat_dgpu (yoloplugin, surface->buf_data[i],
              &rect_params, *yoloplugin->cvmats.at (i), scale_ratio,
              yoloplugin->video_info.width, yoloplugin->video_info.height)
          != GST_FLOW_OK) {
        goto error;
      }
    }
    // Process to get the outputs
    outputs =
        YoloPluginProcess (yoloplugin->yolopluginlib_ctx, yoloplugin->cvmats);

    for (uint k = 0; k < outputs.size (); ++k) {
      if (!outputs.at (k))
        continue;
      // Attach the metadata for the full frame
      attach_metadata_full_frame (yoloplugin, inbuf, scale_ratio,
          outputs.at (k), k);
      free (outputs.at (k));
    }
  } else {
    // Using object crops as input to the algorithm. The objects are detected by
    // the primary detector
    GstMeta *gst_meta;
    NvDsMeta *dsmeta;
    // NOTE: Initializing state to NULL is essential
    gpointer state = NULL;
    NvDsFrameMeta *bbparams;

    // Standard way of iterating through buffer metadata
    while ((gst_meta = gst_buffer_iterate_meta (inbuf, &state)) != NULL) {
      // Check if this metadata is of NvDsMeta type
      if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark))
        continue;

      dsmeta = (NvDsMeta *) gst_meta;
      // Check if the metadata of NvDsMeta contains object bounding boxes
      if (dsmeta->meta_type != NVDS_META_FRAME_INFO)
        continue;

      bbparams = (NvDsFrameMeta *) dsmeta->meta_data;
      // Check if these parameters have been set by the primary detector /
      // tracker
      if (bbparams->gie_type != 1) {
        continue;
      }
      // Iterate through all the objects
      for (guint i = 0; i < bbparams->num_rects; i++) {
        NvDsObjectParams *obj_param = &bbparams->obj_params[i];

        // Crop and scale the object
        if (get_converted_mat_dgpu (yoloplugin,
                surface->buf_data[bbparams->batch_id], &obj_param->rect_params,
                *yoloplugin->cvmats.at (i), scale_ratio,
                yoloplugin->video_info.width, yoloplugin->video_info.height)
            != GST_FLOW_OK) {
          continue;
        }
        if (!obj_param->text_params.display_text) {
          bbparams->num_strings++;
        }
      }
      // Process the object crop to obtain label
      outputs =
          YoloPluginProcess (yoloplugin->yolopluginlib_ctx, yoloplugin->cvmats);

      for (uint k = 0; k < outputs.size (); ++k) {
        if (!outputs.at (k))
          continue;
        NvDsObjectParams *obj_param = &bbparams->obj_params[k];
        // Attach labels for the object
        attach_metadata_object (yoloplugin, obj_param, outputs.at (k));
        free (outputs.at (k));
      }
    }
  }

  flow_ret = GST_FLOW_OK;

error:
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Free the metadata allocated in attach_metadata_full_frame
 */
static void
free_ds_meta (gpointer meta_data)
{
  NvDsFrameMeta *params = (NvDsFrameMeta *) meta_data;
  for (guint i = 0; i < params->num_rects; i++) {
    g_free (params->obj_params[i].text_params.display_text);
  }
  g_free (params->obj_params);
  g_free (params);
}

/**
 * Attach metadata for the full frame. We will be adding a new metadata.
 */
static void
attach_metadata_full_frame (GstYoloPlugin * yoloplugin, GstBuffer * inbuf,
    gdouble scale_ratio, YoloPluginOutput * output, guint batch_id)
{
  NvDsMeta *dsmeta;
  NvDsFrameMeta *bbparams =
      (NvDsFrameMeta *) g_malloc0 (sizeof (NvDsFrameMeta));
  // Allocate an array of size equal to the number of objects detected
  bbparams->obj_params
      =
      (NvDsObjectParams *) g_malloc0 (sizeof (NvDsObjectParams) *
      output->numObjects);
  // Should be set to 3 for custom elements
  bbparams->gie_type = 3;
  // Use HW for overlaying boxes
  bbparams->nvosd_mode = NV_OSD_MODE_GPU;
  bbparams->batch_id = batch_id;
  // Font to be used for label text
  static gchar font_name[] = "Arial";
  GST_DEBUG_OBJECT (yoloplugin, "Attaching metadata %d\n", output->numObjects);
  for (gint i = 0; i < output->numObjects; i++) {
    YoloPluginObject *obj = &output->object[i];
    NvDsObjectParams *obj_param = &bbparams->obj_params[i];
    NvOSD_RectParams & rect_params = obj_param->rect_params;
    NvOSD_TextParams & text_params = obj_param->text_params;

    // Assign bounding box coordinates
    rect_params.left = obj->left;
    rect_params.top = obj->top;
    rect_params.width = obj->width;
    rect_params.height = obj->height;

    // Semi-transparent yellow background
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    // Red border of width 6
    rect_params.border_width = 1;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    // Scale the bounding boxes proportionally based on how the object/frame was
    // scaled during input
    rect_params.left /= scale_ratio;
    rect_params.top /= scale_ratio;
    rect_params.width /= scale_ratio;
    rect_params.height /= scale_ratio;
    GST_DEBUG_OBJECT (yoloplugin,
        "Attaching rect%d of batch%u"
        "  left->%u top->%u width->%u"
        " height->%u label->%s\n",
        i, batch_id, rect_params.left, rect_params.top, rect_params.width,
        rect_params.height, obj->label);
    bbparams->num_rects++;

    // has_new_info should be set to TRUE whenever adding new/updating
    // information to NvDsAttrInfo
    obj_param->has_new_info = TRUE;
    // Update the approriate element of the attr_info array. Application knows
    // that output of this element is available at index "unique_id".
    strcpy (obj_param->attr_info[yoloplugin->unique_id].attr_label, obj->label);
    // is_attr_label should be set to TRUE indicating that above attr_label field is
    // valid
    obj_param->attr_info[yoloplugin->unique_id].is_attr_label = 1;
    // Obj not yet tracked
    obj_param->tracking_id = -1;

    // display_text required heap allocated memory
    text_params.display_text = g_strdup (obj->label);
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    // Font face, size and color
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    bbparams->num_strings++;
  }

  // Attach the NvDsFrameMeta structure as NvDsMeta to the buffer. Pass the
  // function to be called when freeing the meta_data
  dsmeta = gst_buffer_add_nvds_meta (inbuf, bbparams, free_ds_meta);
  dsmeta->meta_type = NVDS_META_FRAME_INFO;
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void
attach_metadata_object (GstYoloPlugin * yoloplugin,
    NvDsObjectParams * obj_param, YoloPluginOutput * output)
{
  if (output->numObjects == 0)
    return;
  NvOSD_TextParams & text_params = obj_param->text_params;
  NvOSD_RectParams & rect_params = obj_param->rect_params;

  // has_new_info should be set to TRUE whenever adding new/updating
  // information to NvDsAttrInfo
  obj_param->has_new_info = TRUE;
  // Update the approriate element of the attr_info array. Application knows
  // that output of this element is available at index "unique_id".
  strcpy (obj_param->attr_info[yoloplugin->unique_id].attr_label,
      output->object[0].label);
  // is_attr_label should be set to TRUE indicating that above attr_label field is
  // valid
  obj_param->attr_info[yoloplugin->unique_id].is_attr_label = 1;
  // Set black background for the text
  // display_text required heap allocated memory
  if (text_params.display_text) {
    gchar *conc_string
        = g_strconcat (text_params.display_text, " ", output->object[0].label,
        NULL);
    g_free (text_params.display_text);
    text_params.display_text = conc_string;
  } else {
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    text_params.display_text = g_strdup (output->object[0].label);
    // Font face, size and color
    text_params.font_params.font_name = "Arial";
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
  }
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
yoloplugin_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_yoloplugin_debug, "yolo", 0, "yolo plugin");

  return gst_element_register (plugin, "nvyolo", GST_RANK_PRIMARY,
      GST_TYPE_YOLOPLUGIN);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, yoloplugin,
    DESCRIPTION, yoloplugin_plugin_init, VERSION, LICENSE, BINARY_PACKAGE, URL)

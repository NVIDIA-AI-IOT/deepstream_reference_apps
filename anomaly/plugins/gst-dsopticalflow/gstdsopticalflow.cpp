////////////////////////////////////////////////////////////////////////////////
// MIT License
// 
// Copyright (C) 2019 NVIDIA CORPORATION
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/time.h>
#include "gstdsopticalflow.h"
#include "nvdsoptflow.h"

//#define DEBUG_OPTF

GST_DEBUG_CATEGORY_STATIC (gst_ds_optical_flow_debug);
#define GST_CAT_DEFAULT gst_ds_optical_flow_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_PROCESSING_WIDTH,
    PROP_PROCESSING_HEIGHT,
    PROP_GPU_DEVICE_ID,
    PROP_POOL_SIZE,
    PROP_ENABLE_HEATMAP

};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 0
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 360
#define DEFAULT_GPU_ID 0
#define DEFAULT_POOL_SIZE 7

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_ds_optical_flow_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12 }")));

static GstStaticPadTemplate gst_ds_optical_flow_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12 }") ";"
                            GST_VIDEO_CAPS_MAKE("{ RGB }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_optical_flow_parent_class parent_class
G_DEFINE_TYPE (GstDsOpticalFlow, gst_ds_optical_flow, GST_TYPE_BASE_TRANSFORM);

static void gst_ds_optical_flow_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_ds_optical_flow_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_ds_optical_flow_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize);

static GstCaps* gst_ds_optical_flow_fixate_caps(GstBaseTransform* btrans,
        GstPadDirection direction, GstCaps* caps, GstCaps* othercaps);

static gboolean gst_ds_optical_flow_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);

static GstCaps* gst_ds_optical_flow_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
    GstCaps* caps, GstCaps* filter);

static gboolean gst_ds_optical_flow_start (GstBaseTransform * btrans);
static gboolean gst_ds_optical_flow_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_ds_optical_flow_transform(GstBaseTransform* btrans, 
    GstBuffer* inbuf, GstBuffer* outbuf);
static GstFlowReturn gst_ds_optical_flow_transform_ip(GstBaseTransform* btrans,
    GstBuffer* inbuf);

static void attach_metadata_object (GstDsOpticalFlow * dsopticalflow,
    NvDsObjectParams * obj_param, DsOpticalFlowOutput * output);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_ds_optical_flow_class_init (GstDsOpticalFlowClass * klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;
    gobject_class = (GObjectClass *) klass;
    gstelement_class = (GstElementClass *) klass;
    gstbasetransform_class = (GstBaseTransformClass *) klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_ds_optical_flow_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_ds_optical_flow_get_property);

    gstbasetransform_class->transform_size = GST_DEBUG_FUNCPTR(gst_ds_optical_flow_transform_size);
    gstbasetransform_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_ds_optical_flow_fixate_caps);
    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_ds_optical_flow_set_caps);
    gstbasetransform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_ds_optical_flow_transform_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_ds_optical_flow_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_ds_optical_flow_stop);

    gstbasetransform_class->transform_ip =
        GST_DEBUG_FUNCPTR (gst_ds_optical_flow_transform_ip);
    gstbasetransform_class->transform =
        GST_DEBUG_FUNCPTR (gst_ds_optical_flow_transform);

    gstbasetransform_class->passthrough_on_same_caps = TRUE;

    /* Install properties */
    g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint ("unique-id",
            "Unique ID",
            "Unique ID for the element. Can be used to identify output of the"
            " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
        g_param_spec_int ("processing-width",
            "Processing Width",
            "Width of the input buffer to algorithm",
            1, G_MAXINT, DEFAULT_PROCESSING_WIDTH, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
       g_param_spec_int ("processing-height",
            "Processing Height",
            "Height of the input buffer to algorithm",
            1, G_MAXINT, DEFAULT_PROCESSING_HEIGHT, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint ("gpu-id",
            "Set GPU Device ID",
            "Set GPU Device ID", 0,
            G_MAXUINT, 0,
            GParamFlags
            (G_PARAM_READWRITE |
                G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property (gobject_class, PROP_POOL_SIZE,
       g_param_spec_uint ("pool-size",
            "Pool Size",
            "Size of optical flow memory pool",
            1, G_MAXUINT, DEFAULT_POOL_SIZE, (GParamFlags)
            (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_ENABLE_HEATMAP, 
        g_param_spec_boolean("enable-heatmap", 
        "Enable Heatmap", 
        "Enable optical flow heatmap output",
        TRUE, G_PARAM_READWRITE));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_ds_optical_flow_src_template));
    gst_element_class_add_pad_template (gstelement_class,
        gst_static_pad_template_get (&gst_ds_optical_flow_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple (gstelement_class,
        "DsOpticalFlow plugin",
        "DsOpticalFlow Plugin",
        "Process a 3rdparty optical flow algorithm on objects / full frame",
        "Chunlin Li<chunlinl@nvidia.com> ");
}

static void
gst_ds_optical_flow_init (GstDsOpticalFlow * dsopticalflow)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsopticalflow);


    /* Initialize all property variables to default values */
    dsopticalflow->unique_id = DEFAULT_UNIQUE_ID;
    dsopticalflow->processing_width = DEFAULT_PROCESSING_WIDTH;
    dsopticalflow->processing_height = DEFAULT_PROCESSING_HEIGHT;
    dsopticalflow->gpu_id = DEFAULT_GPU_ID;
    dsopticalflow->pool_size = DEFAULT_POOL_SIZE;
    dsopticalflow->enable_heatmap = TRUE;
    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
      _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_ds_optical_flow_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstDsOpticalFlow *dsopticalflow = GST_DS_OPTICAL_FLOW (object);
    switch (prop_id) {
      case PROP_UNIQUE_ID:
        dsopticalflow->unique_id = g_value_get_uint (value);
        break;
      case PROP_PROCESSING_WIDTH:
        dsopticalflow->processing_width = g_value_get_int (value);
        break;
      case PROP_PROCESSING_HEIGHT:
        dsopticalflow->processing_height = g_value_get_int (value);
        break;
      case PROP_GPU_DEVICE_ID:
        dsopticalflow->gpu_id = g_value_get_uint (value);
        break;
      case PROP_POOL_SIZE:
        dsopticalflow->pool_size = g_value_get_uint (value);
        break;
      case PROP_ENABLE_HEATMAP:
        dsopticalflow->enable_heatmap = g_value_get_boolean (value);
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
gst_ds_optical_flow_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
    GstDsOpticalFlow *dsopticalflow = GST_DS_OPTICAL_FLOW (object);
    switch (prop_id) {
      case PROP_UNIQUE_ID:
        g_value_set_uint (value, dsopticalflow->unique_id);
        break;
      case PROP_PROCESSING_WIDTH:
        g_value_set_int (value, dsopticalflow->processing_width);
        break;
      case PROP_PROCESSING_HEIGHT:
        g_value_set_int (value, dsopticalflow->processing_height);
        break;
      case PROP_GPU_DEVICE_ID:
        g_value_set_uint (value, dsopticalflow->gpu_id);
        break;
      case PROP_POOL_SIZE:
        g_value_set_uint (value, dsopticalflow->pool_size);
        break;
      case PROP_ENABLE_HEATMAP:
        g_value_set_boolean(value, dsopticalflow->enable_heatmap);
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
gst_ds_optical_flow_start(GstBaseTransform *btrans)
{
    GstDsOpticalFlow *dsopticalflow = GST_DS_OPTICAL_FLOW(btrans);

    /* Algorithm specific initializations and resource allocation. */
    dsopticalflow->dsopticalflowlib_ctx = DsOpticalFlowCreate();
    if (dsopticalflow->dsopticalflowlib_ctx == NULL) return FALSE;

    GST_DEBUG_OBJECT(dsopticalflow, "ctx lib %p \n", dsopticalflow->dsopticalflowlib_ctx);

    return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_ds_optical_flow_stop (GstBaseTransform * btrans)
{
    GstDsOpticalFlow *dsopticalflow = GST_DS_OPTICAL_FLOW (btrans);

    // Deinit the algorithm library
    DsOpticalFlowCtxDeinit (dsopticalflow->dsopticalflowlib_ctx);
    dsopticalflow->dsopticalflowlib_ctx = NULL;

    GST_DEBUG_OBJECT (dsopticalflow, "ctx lib released \n");

    return TRUE;
}

static gboolean 
gst_ds_optical_flow_transform_size(GstBaseTransform* btrans,
        GstPadDirection dir, GstCaps *caps, gsize size, GstCaps* othercaps, gsize* othersize)
{
    gboolean ret = TRUE;
    GstVideoInfo info;

    ret = gst_video_info_from_caps(&info, othercaps);
    if (ret) *othersize = info.size;

    return ret;
}

/* fixate the caps on the other side */
static GstCaps* gst_ds_optical_flow_fixate_caps(GstBaseTransform* btrans,
        GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
    GstDsOpticalFlow* dsopticalflow = GST_DS_OPTICAL_FLOW(btrans);
    GstStructure *s1, *s2;
    GstCaps* result;
    gint num, denom;

    othercaps = gst_caps_truncate(othercaps);
    othercaps = gst_caps_make_writable(othercaps);
    s2 = gst_caps_get_structure(othercaps, 0);
    
    /* wanna trigger transform_ip_passthrough when it is necessary, say the other pad
       is using the same format. However I can't find the right API to do the judge and
       has to intersect it */
    if (gst_caps_can_intersect(othercaps, caps))
    {
        /* the makes the caps on the other side identical to that on this pad, thus
           will do passthrough since we've already set passthrough_on_same_caps */
        result = gst_caps_copy(caps);
    }
    else
    {
        /* otherwise the dimension of the output heatmap needs to be fixated */
        gst_structure_fixate_field_nearest_int(s2, "width", dsopticalflow->processing_width);
        gst_structure_fixate_field_nearest_int(s2, "height", dsopticalflow->processing_height);
        s1 = gst_caps_get_structure(caps, 0);
        if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
        {
            gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);
        }
        result = gst_caps_ref(othercaps);
    }

    gst_caps_unref(othercaps);
    GST_INFO_OBJECT(dsopticalflow, "CAPS fixate: %" GST_PTR_FORMAT ", direction %d", 
                    result, direction);
    return result;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_ds_optical_flow_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
    GstDsOpticalFlow *dsopticalflow = GST_DS_OPTICAL_FLOW (btrans);

    /* Save the input video information, since this will be required later. */
    gst_video_info_from_caps(&dsopticalflow->video_info, incaps);

    DsOpticalFlowInitParams init_params =
        {dsopticalflow->processing_width,
         dsopticalflow->processing_height,
         dsopticalflow->pool_size};

    GstQuery *queryparams = NULL;

    if (cudaSetDevice(dsopticalflow->gpu_id) != cudaSuccess)
    {
        g_printerr("Error: failed to set GPU to %d\n", dsopticalflow->gpu_id);
        return FALSE;
    }

    if (!DsOpticalFlowCtxInit(dsopticalflow->dsopticalflowlib_ctx, &init_params))
    {
        GST_WARNING_OBJECT(dsopticalflow, "Failed to create optical flow context");
        return FALSE;
    }

    return TRUE;
}

static GstCaps*
gst_ds_optical_flow_transform_caps(GstBaseTransform* btrans, GstPadDirection dir,
    GstCaps* caps, GstCaps* filter)
{
    GstCaps* othercaps = NULL;
    GstCaps* tmpl = NULL;

    /* caps of the other side is tied to its template regardless of given caps on
       this pad */
    if (dir == GST_PAD_SINK)
    {
        tmpl = gst_pad_get_pad_template_caps(btrans->srcpad);
    }
    else
    {
        tmpl = gst_pad_get_pad_template_caps(btrans->sinkpad);
    }

    if (filter)
    {
        othercaps = gst_caps_intersect_full(filter, tmpl, GST_CAPS_INTERSECT_FIRST);
    }
    else 
    {
        othercaps = gst_caps_ref(tmpl);
    }

    gst_caps_unref(tmpl);

    return othercaps;
}
/**
 * Free the metadata allocated in attach_metadata_full_frame
 */
static void
free_ds_meta (gpointer meta_data)
{
    DsOpticalFlowMeta *ofmeta = (DsOpticalFlowMeta *) meta_data;
    GstDsOpticalFlow* dsopticalflow = GST_DS_OPTICAL_FLOW(ofmeta->parent);
    DsOpticalFlowFreeData(dsopticalflow->dsopticalflowlib_ctx, ofmeta->data);
    gst_object_unref(ofmeta->parent);
    g_free (meta_data);
}

static GstFlowReturn
gst_ds_optical_flow_transform_internal(GstBaseTransform *btrans,
                                       GstBuffer *inbuf, GstBuffer *outbuf)
{
    GstDsOpticalFlow *dsopticalflow = GST_DS_OPTICAL_FLOW (btrans);
    GstMapInfo in_map_info;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;
    gdouble scale_ratio;
    DsOpticalFlowOutput output;
    NvBufSurface *surface = NULL;
    GstNvStreamMeta *streamMeta = NULL;
    NvDsMeta *dsmeta = NULL;
    DsOpticalFlowMeta *ofmeta = NULL;

    memset (&in_map_info, 0, sizeof (in_map_info));
    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
        g_print ("Error: Failed to map gst buffer\n");
        return GST_FLOW_ERROR;
    }

    do
    {
        if (cudaSetDevice(dsopticalflow->gpu_id) != cudaSuccess)
        {
            g_printerr("Error: failed to set GPU to %d\n", dsopticalflow->gpu_id);
            break;
        }

        surface = (NvBufSurface *) in_map_info.data;
        GST_DEBUG_OBJECT (dsopticalflow,
              "processing %" GST_PTR_FORMAT 
              ", Frame %" G_GUINT64_FORMAT " Surface %p",
              inbuf, dsopticalflow->frame_num, surface);

        if (CHECK_NVDS_MEMORY_AND_GPUID (dsopticalflow, surface))
            break;

        output.rgb = NULL;
        output.data = NULL;
        GstMapInfo info;
        /* the optical flow heat map calculation will be skipped in case the plugin 
           works in passthrough mode or enable_heatmap has been set to false */
        if (outbuf)
        {
            gst_buffer_map(outbuf, &info, GST_MAP_WRITE);
            if (dsopticalflow->enable_heatmap)
            {
                output.rgb = info.data;
            }
        }

        // Process to get the output if optical flow is generated correctly
        if (DsOpticalFlowProcess(dsopticalflow->dsopticalflowlib_ctx, surface->buf_data[0], 
                                 dsopticalflow->video_info.width, dsopticalflow->video_info.height,
                                 &output) &&
            output.data != NULL)
        {
            ofmeta = (DsOpticalFlowMeta *)g_malloc0(sizeof(DsOpticalFlowMeta));
            ofmeta->stream_id = dsopticalflow->unique_id;
            ofmeta->frame_num = dsopticalflow->frame_num;
            ofmeta->cols = output.cols;
            ofmeta->rows = output.rows;
            ofmeta->data = output.data;
            ofmeta->elem_size = output.elemSize;
            ofmeta->parent = (GstElement *)dsopticalflow;
            gst_object_ref(ofmeta->parent);

            /* only attach the meta to output buffer when there is one, in passthrough
                   mode we only get input buffer */
            GstBuffer *buf = outbuf ? outbuf : inbuf;
            dsmeta = gst_buffer_add_nvds_meta(buf, ofmeta, free_ds_meta);
            dsmeta->meta_type = NVDS_META_OPTICAL_FLOW;

            GST_DEBUG_OBJECT(dsopticalflow,
                             "optical flow meta attached to %" GST_PTR_FORMAT, buf);
#if defined(DEBUG_OPTF)
            if (output.rgb)
            {
                cv::Mat cvrgb(output.rows, output.cols, CV_8UC3, output.rgb);
                char f[50];
                std::vector<int> params;
                params.push_back(cv::IMWRITE_JPEG_QUALITY);
                params.push_back(80);
                sprintf(f, "result%u_%lu.jpg", dsopticalflow->unique_id, dsopticalflow->frame_num);
                imwrite(f, cvrgb, params);
            }
#endif
        }

        if (outbuf) gst_buffer_unmap(outbuf, &info);

        flow_ret = GST_FLOW_OK;
        dsopticalflow->frame_num++;
    } while (0);

    gst_buffer_unmap (inbuf, &in_map_info);
    return flow_ret;
}

/**
 * Called when the plugin works in non-passthough mode
 */
static GstFlowReturn
gst_ds_optical_flow_transform(GstBaseTransform* btrans, GstBuffer* inbuf, GstBuffer* outbuf)
{
    return gst_ds_optical_flow_transform_internal(btrans, inbuf, outbuf);
}

/**
 * Called when the plugin works in passthough mode
 */
static GstFlowReturn
gst_ds_optical_flow_transform_ip(GstBaseTransform* btrans, GstBuffer* inbuf)
{
    return gst_ds_optical_flow_transform_internal(btrans, inbuf, NULL);
}


/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
ds_optical_flow_plugin_init (GstPlugin * plugin)
{
    GST_DEBUG_CATEGORY_INIT (gst_ds_optical_flow_debug, "dsopticalflow", 0,
      "dsopticalflow plugin");

    return gst_element_register (plugin, "dsopticalflow", GST_RANK_PRIMARY,
          GST_TYPE_DS_OPTICAL_FLOW);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    dsopticalflow,
    DESCRIPTION, ds_optical_flow_plugin_init, "3.0", LICENSE, BINARY_PACKAGE, URL)

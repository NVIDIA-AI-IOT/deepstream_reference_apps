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
#include "gstdsdirection.h"
#include <gst/video/video.h>
#include "gstnvdsmeta.h"
#include "gstnvstreammeta.h"
#include "../gst-dsopticalflow/nvdsoptflow.h"
#include "dsdirection_lib/dsdirection_lib.h"
#include <string.h>

GST_DEBUG_CATEGORY_STATIC(gst_ds_direction_debug);
#define GST_CAT_DEFAULT gst_ds_direction_debug

/* Package and library details required for plugin_init */
#define PACKAGE "dsdirection"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA direction estimation plugin for integration with DeepStream on DGPU"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty IP integration opticalflow plugin"
#define URL "http://nvidia.com/"

#define NUM_SINK_LIMIT 2

/* Filter signals and args */
enum
{
    /* FILL ME */
    LAST_SIGNAL
};

enum
{
    PROP_0,
    PROP_SILENT
};

enum
{
    SINK_MAIN,
    SINK_OPTICAL_FLOW,
    SINK_INVALID
};

enum
{
    CLASS_VEHICLE,
    CLASS_BICYCLE,
    CLASS_PERSON,
    CLASS_ROADSIGN
};

static GQuark _dsmeta_quark = 0;

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12 }")));

static GstStaticPadTemplate optf_sink_factory =
    GST_STATIC_PAD_TEMPLATE("optf_sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE("{RGB}")));

static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_REQUEST,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                            "memory:NVMM",
                            "{ NV12 }") ";"
                            GST_VIDEO_CAPS_MAKE("{RGB}")));

#define gst_ds_direction_parent_class parent_class
G_DEFINE_TYPE(GstDsDirection, gst_ds_direction, GST_TYPE_ELEMENT);

/* Callbacks */
static GstFlowReturn gst_ds_direction_collected(GstCollectPads *pads, gpointer user_data);
static gboolean gst_ds_direction_sink_event(GstCollectPads *pads, GstCollectData *data,
                                         GstEvent *event, gpointer user_data);

/* GObject vmethods */
static void gst_ds_direction_set_property(GObject *object, guint prop_id,
                                       const GValue *value, GParamSpec *pspec);
static void gst_ds_direction_get_property(GObject *object, guint prop_id,
                                       GValue *value, GParamSpec *pspec);

/* GstElement vmethods */
static GstStateChangeReturn gst_ds_direction_change_state(GstElement *element,
                                                       GstStateChange transition);
static GstPad *gst_ds_direction_request_src_pad(GstElement *element, GstPadTemplate *templ,
                                           const gchar *name, const GstCaps *caps);
static void gst_ds_direction_release_src_pad(GstElement *element, GstPad *pad);


/* initialize the plugin's class */
static void
gst_ds_direction_class_init(GstDsDirectionClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;

    gobject_class->set_property = gst_ds_direction_set_property;
    gobject_class->get_property = gst_ds_direction_get_property;

    gstelement_class->change_state = GST_DEBUG_FUNCPTR(gst_ds_direction_change_state);
    gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(gst_ds_direction_request_src_pad);
    gstelement_class->release_pad = GST_DEBUG_FUNCPTR(gst_ds_direction_release_src_pad);

    g_object_class_install_property(gobject_class, PROP_SILENT,
                                    g_param_spec_boolean("silent", "Silent", "Produce verbose output ?",
                                                         FALSE, G_PARAM_READWRITE));

    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&src_factory));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&sink_factory));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&optf_sink_factory));
    gst_element_class_set_details_simple(gstelement_class,
                                         "DsDirection plugin",
                                         "DsDirection Plugin",
                                         "Perform direction estimation on detected objects based on dense optical flow",
                                         "Chunlin Li <chunlinl@nvidia.com> ");
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad callback functions
 * initialize instance structure
 */
static void
gst_ds_direction_init(GstDsDirection *direction)
{
    direction->srcpad = gst_pad_new_from_static_template(&src_factory, "src");
    gst_element_add_pad(GST_ELEMENT(direction), direction->srcpad);

    direction->sinkpads[SINK_MAIN] =
        gst_pad_new_from_static_template(&sink_factory, "sink");
    gst_element_add_pad(GST_ELEMENT(direction), direction->sinkpads[SINK_MAIN]);

    direction->sinkpads[SINK_OPTICAL_FLOW] =
        gst_pad_new_from_static_template(&optf_sink_factory, "optf_sink");
    gst_element_add_pad(GST_ELEMENT(direction), direction->sinkpads[SINK_OPTICAL_FLOW]);


    direction->collect = gst_collect_pads_new();
    gst_collect_pads_set_function(direction->collect,
                                  (GstCollectPadsFunction)GST_DEBUG_FUNCPTR(gst_ds_direction_collected), direction);
    gst_collect_pads_set_event_function(direction->collect,
                                        (GstCollectPadsEventFunction)GST_DEBUG_FUNCPTR(gst_ds_direction_sink_event), direction);

    gst_collect_pads_add_pad(direction->collect, direction->sinkpads[SINK_MAIN], 
                 sizeof(GstCollectData), NULL, TRUE);
    gst_collect_pads_add_pad(direction->collect, direction->sinkpads[SINK_OPTICAL_FLOW], 
                 sizeof(GstCollectData), NULL, TRUE);

    direction->silent = TRUE;
    direction->active_passthru = SINK_INVALID;

    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

static void
gst_ds_direction_set_property(GObject *object, guint prop_id,
                           const GValue *value, GParamSpec *pspec)
{
    GstDsDirection *direction = GST_DS_DIRECTION(object);

    switch (prop_id)
    {
    case PROP_SILENT:
        direction->silent = g_value_get_boolean(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_ds_direction_get_property(GObject *object, guint prop_id,
                           GValue *value, GParamSpec *pspec)
{
    GstDsDirection *direction = GST_DS_DIRECTION(object);

    switch (prop_id)
    {
    case PROP_SILENT:
        g_value_set_boolean(value, direction->silent);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static const char *dirlabel[] =
{
        "",
        "\u21D2", /*< right*/
        "\u21D8", /*< bottom-right */
        "\u21D3", /*< downward */
        "\u21D9", /*< bottom-left */
        "\u21D0", /*< left */
        "\u21D6", /*< top-left */
        "\u21D1", /*< upward */
        "\u21D7" /*< top-right */
};

static float intervals[8][2] =
{
    {7/8.0, -7/8.0},
    {-7/8.0, -5/8.0},
    {-5/8.0, -3/8.0},
    {-3/8.0, -1/8.0},
    {-1/8.0, 1/8.0},
    {1/8.0, 3/8.0},
    {3/8.0, 5/8.0},
    {5/8.0, 7/8.0},
};

static gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

/* internel method: attach direction information to the NvDsFrameMeta structure */
static void
attach_nvmeta(NvDsFrameMeta* bbparams, DsMotionObject* directions)
{
    if (bbparams == NULL || directions == NULL) return;

    float maxrad = 0.0;
    for (guint i = 0; i < bbparams->num_rects; i++)
    {
        if (directions[i].radius > maxrad)
            maxrad = directions[i].radius;
    }

    char label[64];
    float threshold = maxrad/20;
    bbparams->num_strings = 0;
    for (guint i = 0; i < bbparams->num_rects; i++)
    {
        NvDsObjectParams* obj_param = &bbparams->obj_params[i];
        NvOSD_RectParams rect = obj_param->rect_params;
        guint dir = 0;

        g_strlcpy(label, pgie_classes_str[obj_param->class_id], 64);

        /* We want to filter out some objects with minor direction */
        if (directions[i].radius > threshold)
        {
            for (guint d = 1; d < 8; d++)
            {
                if (directions[i].angle > intervals[d][0] && 
                    directions[i].angle <= intervals[d][1])
                {
                    dir = d + 1;
                    break;
                }
            }

            /* special case for right */
            if (dir == 0 &&
                (directions[i].angle > intervals[0][0] || directions[i].angle <= intervals[0][1]))
            {
                dir = 1;
            }
        }

        /* 
         * To-dos: there has to be some field specific for direction indication
         * text is only informal
         */

        // Display text above the left top corner of the object
        obj_param->text_params.x_offset = rect.left;
        obj_param->text_params.y_offset = rect.top - 10;
        // Font face, size and color
        obj_param->text_params.font_params.font_name = "Arial";
        obj_param->text_params.font_params.font_size = 12;
        obj_param->text_params.font_params.font_color = (NvOSD_ColorParams){
            100, 1, 1, 1};
        // Set black background for the text
        obj_param->text_params.set_bg_clr = 1;
        obj_param->text_params.text_bg_clr = (NvOSD_ColorParams){
            0, 0, 0, 1};
        if (obj_param->class_id != CLASS_ROADSIGN)
        {
            g_strlcat(label, dirlabel[dir], 64);
        }
        obj_param->text_params.display_text = g_strdup(label);
        bbparams->num_strings++;
    }
}

/* internal method: direction estimation of detected objects based on dense optical flow */ 
static gboolean gst_ds_direction_estimate(GstDsDirection *ds_direction,
                                         GstBuffer *optflow, GstBuffer *frame)
{
    NvDsMeta* dsmeta = NULL;
    DsOpticalFlowMeta* ofmeta = NULL;
    GstVideoMeta* vidmeta = NULL;
    GstCaps *caps = NULL;
    GstVideoInfo info;

    /* We want to get the dimention of original video frame in order to calculate the
       scaling during direction estimation, since we didn't use VideoFrame to carry the 
       picture we have to read the width and height from CAPS of the pad */
    caps = gst_pad_get_current_caps(ds_direction->sinkpads[SINK_MAIN]);
    gst_video_info_from_caps(&info, caps);
    gst_caps_unref(caps);

    gint32 pic_width = info.width;
    gint32 pic_height = info.height;
    dsmeta = gst_buffer_get_nvds_meta(optflow);
    if (dsmeta && dsmeta->meta_type == NVDS_META_OPTICAL_FLOW)
    {
        ofmeta = (DsOpticalFlowMeta*) dsmeta->meta_data;
        DsOpticalFlowMap ofmap;
        ofmap.cols = ofmeta->cols;
        ofmap.rows = ofmeta->rows;
        ofmap.elemSize = ofmeta->elem_size;
        ofmap.data = ofmeta->data;
        float maxrad = DsDirectionFindMaxRad(&ofmap);

        /* interate the detected objects */
        GstMeta *gst_meta;
        gpointer state = NULL;
        NvDsFrameMeta *bbparams = NULL;
        while ((gst_meta = gst_buffer_iterate_meta(frame, &state)) != NULL)
        {
            if (!gst_meta_api_type_has_tag(gst_meta->info->api, _dsmeta_quark))
                continue;

            dsmeta = (NvDsMeta*) gst_meta;
            if (dsmeta->meta_type != NVDS_META_FRAME_INFO)
                continue;

            bbparams = (NvDsFrameMeta*) dsmeta->meta_data;
            GST_DEBUG_OBJECT(ds_direction, "found bounding boxes from frame %d in stream %d",
                             bbparams->frame_num, bbparams->stream_id);

            if (bbparams->stream_id != ofmeta->stream_id)
            {
                /* only want to perform direction estimation on the video stream from which
                   the optical flow data comes */
                continue;
            }

            DsMotionObject directions[bbparams->num_rects];
            for (guint i = 0;  i < bbparams->num_rects; i++)
            {
                NvDsObjectParams* obj_param = &bbparams->obj_params[i];
                NvOSD_RectParams rect = obj_param->rect_params;

                /* The direction area should be scaled in proportion to the size of 
                   optical flow map, which normally has been scaled down from the 
                   orignial picture */
                directions[i].left = rect.left * ofmap.cols / pic_width;
                directions[i].top = rect.top * ofmap.rows / pic_height;
                directions[i].width = rect.width * ofmap.cols / pic_width;
                directions[i].height = rect.height * ofmap.rows / pic_height;
                /* output as {radius, angle} */
                directions[i].radius = 0;
                directions[i].angle = 0;
                DsDirectionEstimation(&ofmap, &directions[i], maxrad);

                GST_LOG_OBJECT(ds_direction,
                                 "Direction detected on Object {%d, %d, %d, %d}, angle=%.2f, strength=%.2f",
                                 rect.left, rect.top, rect.width, rect.height,
                                 directions[i].angle, directions[i].radius);
            }

            /* add direction tags to the bounding boxes */
            attach_nvmeta(bbparams, directions);
        }
    }

    return TRUE;
}

static gboolean gst_ds_direction_collect_buffer(GstDsDirection* direction, GstCollectPads *collect, 
                                             GstCollectData **optflow_data, 
                                             GstCollectData **nvframe_data)
{
    gboolean eos = FALSE;
    GstBuffer *optflow = NULL;
    GstBuffer *nvframe = NULL;

    GSList *walk = collect->data;
    while (walk)
    {
        GstCollectData *data = (GstCollectData *)walk->data;
        walk = g_slist_next(walk);
        if (GST_COLLECT_PADS_STATE_IS_SET(data, GST_COLLECT_PADS_STATE_EOS))
        {
            GST_INFO_OBJECT(direction, "pad %" GST_PTR_FORMAT "goes eos", data->pad);
            eos = TRUE;
            continue;
        }

        eos = FALSE;
        GstBuffer *buf = gst_collect_pads_peek(collect, data);
        if (buf)
        {
            if (data->pad == direction->sinkpads[SINK_MAIN])
            {
                nvframe = buf;
                *nvframe_data = data;
            }
            else if (data->pad == direction->sinkpads[SINK_OPTICAL_FLOW])
            {
                optflow = buf;
                *optflow_data = data;
            }
        }
    }

    if (*nvframe_data && *optflow_data)
    {
        gint64 optflow_frame = -1;
        gint64 nvframe_frame = -1;
        guint stream_id = (guint)-1;

        /* find the stream id and frame number from optical flow meta first */
        NvDsMeta *dsmeta = gst_buffer_get_nvds_meta(optflow);
        if (dsmeta && dsmeta->meta_type == NVDS_META_OPTICAL_FLOW)
        {
            DsOpticalFlowMeta *ofmeta = (DsOpticalFlowMeta*) dsmeta->meta_data;
            optflow_frame = ofmeta->frame_num;
            stream_id = ofmeta->stream_id;
        }

        GST_DEBUG_OBJECT(direction, "found optical flow data for frame %" G_GUINT64_FORMAT 
                         " in stream %u", optflow_frame, stream_id);

        GstNvStreamMeta *streamMeta = gst_buffer_get_nvstream_meta(nvframe);
        if (streamMeta)
        {
            for (guint i = 0; i < streamMeta->num_filled; i++)
            {
                if (streamMeta->stream_id[i] == stream_id)
                {
                    nvframe_frame = streamMeta->stream_frame_num[i];
                    GST_DEBUG_OBJECT(direction, "found frame data for frame %" G_GUINT64_FORMAT 
                                     " in stream %u", nvframe_frame, stream_id);
                    break;
                }
            }
        }

        /* Here we use a simple frame to frame method to synchronize the optical flow 
           data and inferred video frame: if one comes faster, the slower one will be dropped
           and no further processing will be done */
        if (nvframe_frame > optflow_frame)
        {
            GST_WARNING_OBJECT(direction, 
                               "optical flow data lagged behind by %u frames, dropping",
                               (guint)(nvframe_frame - optflow_frame));
            gst_buffer_unref(optflow);
            optflow = gst_collect_pads_pop(collect, *optflow_data);
            *optflow_data = NULL;
        }
        else if (nvframe_frame < optflow_frame)
        {
            GST_WARNING_OBJECT(direction, 
                               "frame data lagged behind by %u frames, dropping",
                               (guint)(optflow_frame - nvframe_frame));
            gst_buffer_unref(nvframe);
            nvframe = gst_collect_pads_pop(collect, *nvframe_data);
            *nvframe_data = NULL;
        }
    }

    if (nvframe) gst_buffer_unref(nvframe);
    if (optflow) gst_buffer_unref(optflow);

    return eos;
}

/* The callback for collectpads to trigger checking the arriving data
   If all the pads are in EOS, the func should return GST_FLOW_EOS
 */
static GstFlowReturn gst_ds_direction_collected(GstCollectPads *collect, gpointer user_data)
{
    GstFlowReturn ret = GST_FLOW_OK;
    GstDsDirection *ds_direction = (GstDsDirection *)user_data;
    gboolean eos = TRUE;
    GstCollectData *optflow_data = NULL;
    GstCollectData *nvframe_data = NULL;
    GstBuffer* optflow = NULL;
    GstBuffer* nvframe = NULL;
    GstBuffer* outbuf = NULL;


    eos = gst_ds_direction_collect_buffer(ds_direction, collect, &optflow_data, &nvframe_data);
    if (eos)
    {
        return GST_FLOW_EOS;
    }
   
    /* we got valid optical flow data and inferred video frame, do estimation now */
    if (optflow_data && nvframe_data)
    {
        optflow = gst_collect_pads_pop(collect, optflow_data);
        nvframe = gst_collect_pads_pop(collect, nvframe_data);

        gst_ds_direction_estimate(ds_direction, optflow, nvframe);
        if (ds_direction->active_passthru == SINK_MAIN)
        {
            gst_buffer_unref(optflow);
            outbuf = nvframe;
        }
        else if (ds_direction->active_passthru == SINK_OPTICAL_FLOW)
        {
            gst_buffer_unref(nvframe);
            outbuf = optflow;
        }
        else /*< no active passthru, simply drop both the buffers */
        {
            gst_buffer_unref(optflow);
            gst_buffer_unref(nvframe);
            outbuf = NULL;
        }
    }

    if (outbuf)
    {
        gst_pad_push(ds_direction->srcpad, outbuf);
        if (!ds_direction->silent)
        {
            g_print("Direction estimated and output buffer pushed downstream\n");
        }
    }

    return ret;
}

static gboolean gst_ds_direction_sink_setcaps(GstDsDirection *direction, GstPad* pad, GstCaps* caps)
{
    gboolean ret = TRUE;
    gboolean accepted;
    GstCaps* outcaps = NULL;
    GstCaps* prevcaps = NULL;

    GST_DEBUG_OBJECT(direction, "new caps %" GST_PTR_FORMAT " comes from pad %" GST_PTR_FORMAT,
                     caps, pad);

    if (direction->srcpad)
    {
        outcaps = gst_pad_peer_query_caps(direction->srcpad, caps);
        if (outcaps == NULL || gst_caps_is_empty(outcaps))
        {
            GST_INFO_OBJECT(direction, "caps from pad %" GST_PTR_FORMAT " is not compatible",
                            pad);
            ret = false;
        }
        else
        {
            ret = gst_pad_set_caps(direction->srcpad, outcaps);
        }

        /* the caps has been successfully set to downstream components, so let's
           decide the passthrough mode */
        if (ret && direction->active_passthru == SINK_INVALID)
        {
            if (pad == direction->sinkpads[SINK_MAIN])
            {
                direction->active_passthru = SINK_MAIN;
                GST_INFO_OBJECT(direction, "active passthrough on sink main");
            }
            else if (pad == direction->sinkpads[SINK_OPTICAL_FLOW])
            {
                direction->active_passthru = SINK_OPTICAL_FLOW;
                GST_INFO_OBJECT(direction, "active passthrough on sink opticalflow");
            }
        }
        if (outcaps)  gst_caps_unref(outcaps);
    }

    return ret;
}

static gboolean gst_ds_direction_sink_event(GstCollectPads *pads, GstCollectData *data,
                                         GstEvent *event, gpointer user_data)
{
    GstDsDirection *ds_direction = (GstDsDirection *)user_data;
    gboolean ret;
    gboolean forward = TRUE;
    gboolean discard = FALSE;

    switch (GST_EVENT_TYPE(event))
    {
    case GST_EVENT_EOS:
        GST_INFO_OBJECT(ds_direction, "reaching eos");
        gst_pad_push_event(ds_direction->srcpad, event);
        break;
    case GST_EVENT_CAPS:
    {
        GstCaps *caps;
        gst_event_parse_caps(event, &caps);
        if (!gst_ds_direction_sink_setcaps(ds_direction, data->pad, caps))
        {
            GST_INFO_OBJECT(ds_direction, "failed to do setcaps on pad %" GST_PTR_FORMAT,
                            data->pad);
            gst_event_unref(event);
        }
        forward = FALSE;
    }
        
        break;
    case GST_EVENT_SINK_MESSAGE:
    {
        /* We only want to distribute the sink message event from the pad that is set
           as the passthrough source */
        if (ds_direction->active_passthru == SINK_INVALID ||
           (ds_direction->active_passthru == SINK_OPTICAL_FLOW && data->pad == ds_direction->sinkpads[SINK_MAIN]) ||
           (ds_direction->active_passthru == SINK_MAIN && data->pad == ds_direction->sinkpads[SINK_OPTICAL_FLOW]))
           {
                discard = TRUE;
           }
        break;
    }
    default:
        break;
    }

    if (forward)
    {
        ret = gst_collect_pads_event_default(pads, data, event, discard);
    }      
    
    return ret;
}

static GstStateChangeReturn gst_ds_direction_change_state(GstElement *element,
                                                       GstStateChange transition)
{
    GstStateChangeReturn ret;
    GstDsDirection *ds_direction = GST_DS_DIRECTION(element);

    switch (transition)
    {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
        gst_collect_pads_start(ds_direction->collect);
        break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
        gst_collect_pads_stop(ds_direction->collect);
        break;
    default:
        break;
    }

    ret = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);

    return ret;
}

static GstPad *gst_ds_direction_request_src_pad(GstElement *element, GstPadTemplate *templ,
                                             const gchar *name, const GstCaps *caps)
{
    GstPad* pad = NULL;
    GstDsDirection *direction = GST_DS_DIRECTION(element);

    if (direction->srcpad == NULL)
    {
        pad = gst_pad_new_from_template(templ, name);
        if (pad)
        {
            gst_element_add_pad(element, pad);
            direction->srcpad = pad;

            GST_INFO_OBJECT(direction, "requested a source pad %" GST_PTR_FORMAT, pad);
        }
    }

    return pad;
}

static void gst_ds_direction_release_src_pad(GstElement *element, GstPad *pad)
{
    GstDsDirection *direction = GST_DS_DIRECTION(element);

    gst_object_ref(pad);
    gst_element_remove_pad(element, pad);
    gst_pad_set_active(pad, FALSE);
    gst_object_unref(pad);

    direction->srcpad = NULL;
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
plugin_init(GstPlugin *plugin)
{
    /* debug category for fltering log messages
   *
   * exchange the string 'DS direction plugin' with your description
   */
    GST_DEBUG_CATEGORY_INIT(gst_ds_direction_debug, "dsdirection",
                            0, "DS direction plugin");

    return gst_element_register(plugin, "dsdirection", GST_RANK_NONE,
                                GST_TYPE_DS_DIRECTION);
}

/* gstreamer looks for this structure to register plugins
 *
 * exchange the string 'DS direction plugin' with your plugin description
 */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    dsdirection,
    DESCRIPTION,
    plugin_init,
    VERSION,
    LICENSE,
    PACKAGE,
    URL)

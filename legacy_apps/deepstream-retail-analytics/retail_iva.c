/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <sys/timeb.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvds_yml_parser.h"

#define MAX_DISPLAY_LEN 64
#define MAX_TIME_STAMP_LEN 32

// Config files for detectors, tracker and message broker
#define PGIE_CONFIG_FILE "configs/pgie_config_peoplenet.txt"
#define SGIE_CONFIG_FILE "configs/basket_classifier.txt"
#define TRACKER_CONFIG_FILE "configs/dstest4_tracker_config.txt"
#define MSCONV_CONFIG_FILE "configs/dstest4_msgconv_config.txt"

// Primary detector class IDs
#define PGIE_CLASS_ID_PERSON 0
#define PGIE_CLASS_ID_FACE 1
#define PGIE_CLASS_ID_BAG 2

// Properties of nvstreammux
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define MUXER_BATCH_TIMEOUT_USEC 40000

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

// Keys to read tracker config file
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"

// Define variables to store config parameters
static gchar *cfg_file = NULL;
static gchar *input_file = NULL;
static gchar *topic = NULL;
static gchar *conn_str = NULL;
static gchar *proto_lib = NULL;
static gint schema_type = 0;
static gint msg2p_meta = 0;
static gint frame_interval = 15;
static gboolean display_off = FALSE;

// Array to store classes detected by PGIE
gchar pgie_classes_str[3][32] = {"Person", "Bag", "Face"};

GOptionEntry entries[] = {
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME, &cfg_file,
      "Set the adaptor config file. Optional if connection string has relevant  details.",
        NULL},
  {"input-file", 'i', 0, G_OPTION_ARG_FILENAME, &input_file,
      "Set the input H264 file", NULL},
  {"topic", 't', 0, G_OPTION_ARG_STRING, &topic,
      "Name of message topic. Optional if it is part of connection string or config file.",
        NULL},
  {"conn-str", 0, 0, G_OPTION_ARG_STRING, &conn_str,
      "Connection string of backend server. Optional if it is part of config file.",
        NULL},
  {"proto-lib", 'p', 0, G_OPTION_ARG_STRING, &proto_lib,
      "Absolute path of adaptor library", NULL},
  {"schema", 's', 0, G_OPTION_ARG_INT, &schema_type,
      "Type of message schema (0=Full, 1=minimal), default=0", NULL},
  {"msg2p-meta", 0, 0, G_OPTION_ARG_INT, &msg2p_meta,
      "msg2payload generation metadata type (0=Event Msg meta, 1=nvds meta), default=0",
        NULL},
  {"frame-interval", 0, 0, G_OPTION_ARG_INT, &frame_interval,
      "Frame interval at which payload is generated , default=30", NULL},
  {"no-display", 0, 0, G_OPTION_ARG_NONE, &display_off, "Disable display",
        NULL},
  {NULL}
};

// Function to check if an input file is a YAML file
#define IS_YAML(file) (g_str_has_suffix (file, ".yml") || g_str_has_suffix (file, ".yaml"))

gint frame_number = 0;

// Function to generate timestamp for kafka message
static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6];              //.nnnZ\0

    clock_gettime (CLOCK_REALTIME, &ts);
    memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
    gmtime_r (&tloc, &tm_log);
    strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
    strncat (buf, strmsec, buf_size);
}

// Callback function for deep-copying NvDsEventMsgMeta struct
static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = g_memdup (srcMeta, sizeof (NvDsEventMsgMeta));

    if (srcMeta->ts)
        dstMeta->ts = g_strdup (srcMeta->ts);

    if (srcMeta->sensorStr)
        dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

    if (srcMeta->objSignature.size > 0) {
        dstMeta->objSignature.signature = g_memdup (srcMeta->objSignature.signature,
            srcMeta->objSignature.size);
        dstMeta->objSignature.size = srcMeta->objSignature.size;
    }

    if (srcMeta->objectId) {
        dstMeta->objectId = g_strdup (srcMeta->objectId);
    }

    if (srcMeta->extMsgSize > 0) {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
        NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
        NvDsPersonObject *obj =
            (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

        if (srcObj->hasBasket)
            obj->hasBasket = g_strdup (srcObj->hasBasket);

        dstMeta->extMsg = obj;
        dstMeta->extMsgSize = sizeof (NvDsPersonObject);
        }
    }

    return dstMeta;
}

// Callback function to free NvDsEventMsgMeta struct
static void
meta_free_func (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

    g_free (srcMeta->ts);
    g_free (srcMeta->sensorStr);

    if (srcMeta->objSignature.size > 0) {
        g_free (srcMeta->objSignature.signature);
        srcMeta->objSignature.size = 0;
    }

    if (srcMeta->objectId) {
        g_free (srcMeta->objectId);
    }

    if (srcMeta->extMsgSize > 0) {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
        NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

        if (obj->hasBasket)
            g_free (obj->hasBasket);
        }
        g_free (srcMeta->extMsg);
        srcMeta->extMsgSize = 0;
    }
    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}

// Function to return label from classifier metadata
static gchar *
get_first_result_label (NvDsClassifierMeta * classifierMeta)
{
    GList *n;
    // Iterate through all the secondary labels stored in classifierMeta
    // Refer to deepstream-test5
    for (n = classifierMeta->label_info_list; n != NULL; n = n->next) {
        NvDsLabelInfo *labelInfo = (NvDsLabelInfo *) (n->data);
        if (labelInfo->result_label[0] != '\0') {
            return g_strdup (labelInfo->result_label);
        }
    }
    return NULL;
}

// Function to generate metadata for person object type
static void
generate_person_meta (gpointer data)
{
    NvDsPersonObject *obj = (NvDsPersonObject *) data;
}

// Function to generate event message metadata from object metadata
static void
generate_event_msg_meta (gpointer data, gint class_id,
    NvDsObjectMeta * obj_params)
{
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
    meta->sensorId = 0;
    meta->placeId = 0;
    meta->moduleId = 0;
    meta->sensorStr = g_strdup ("sensor-0");

    meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
    meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

    strncpy (meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

    generate_ts_rfc3339 (meta->ts, MAX_TIME_STAMP_LEN);

    // * This demonstrates how to attach custom objects.
    // * Any custom object as per requirement can be generated and attached
    // * like NvDsVehicleObject / NvDsPersonObject. Then that object should
    // * be handled in payload generator library (nvmsgconv.cpp) accordingly.    
    
    // Attach the secondary label to the person object detected

    if (class_id == PGIE_CLASS_ID_PERSON) {
        meta->type = NVDS_EVENT_ENTRY;
        meta->objType = NVDS_OBJECT_TYPE_PERSON;
        meta->objClassId = PGIE_CLASS_ID_PERSON;

        NvDsPersonObject *obj =
            (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
        generate_person_meta (obj);
        
        GList *l;
        for (l = obj_params->classifier_meta_list; l!= NULL; l = l->next) {
            NvDsClassifierMeta *classifierMeta = (NvDsClassifierMeta *) (l->data);
            obj->hasBasket = get_first_result_label(classifierMeta);
        }

        meta->extMsg = obj;
        meta->extMsgSize = sizeof (NvDsPersonObject);
    }
}

// Probe function to generate OSD data
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    // Uncomment lines containing is_first_object send kafka messages
    // for only the first detection in the frame.

    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameMeta *frame_meta = NULL;
    NvOSD_TextParams *txt_params = NULL;
    guint person_count = 0;
    // gboolean is_first_object = TRUE;
    NvDsMetaList *l_frame, *l_obj;
    gchar *sgie_label=NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    if (!batch_meta) {
        // No batch meta attached.
        return GST_PAD_PROBE_OK;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        frame_meta = (NvDsFrameMeta *) l_frame->data;

        if (frame_meta == NULL) {
            // Ignore Null frame meta.
            continue;
        }

        // is_first_object = TRUE;

        for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            if (obj_meta == NULL) {
                // Ignore Null object.
                continue;
            }

            GList *l;
            sgie_label = "NULL";
            for (l = obj_meta->classifier_meta_list; l!= NULL; l = l->next) {
                NvDsClassifierMeta *classifierMeta = (NvDsClassifierMeta *) (l->data);
                sgie_label = get_first_result_label(classifierMeta);
            }

            txt_params = &(obj_meta->text_params);
            if (txt_params->display_text)
            g_free (txt_params->display_text);

            txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);

            g_snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "%s %ld %s",
                pgie_classes_str[obj_meta->class_id], obj_meta->object_id, sgie_label); /* Person 12 hasBasket */

            person_count++;

            /* Now set the offsets where the string should appear */
            txt_params->x_offset = obj_meta->rect_params.left;
            txt_params->y_offset = obj_meta->rect_params.top - 25;

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

            // * Ideally NVDS_EVENT_MSG_META should be attached to buffer by the
            // * component implementing detection / recognition logic.
            // * Here it demonstrates how to use / attach that meta data.

            if (/*is_first_object && */ !(frame_number % frame_interval)) {
                /* Frequency of messages to be send will be based on use case.
                * Here message is being sent for first object every frame_interval(default=15).
                */
                if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                    NvDsEventMsgMeta *msg_meta =
                        (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
                    msg_meta->bbox.top = obj_meta->rect_params.top;
                    msg_meta->bbox.left = obj_meta->rect_params.left;
                    msg_meta->bbox.width = obj_meta->rect_params.width;
                    msg_meta->bbox.height = obj_meta->rect_params.height;
                    msg_meta->frameId = frame_number;
                    msg_meta->trackingId = obj_meta->object_id;
                    msg_meta->confidence = obj_meta->confidence;
                    generate_event_msg_meta (msg_meta, obj_meta->class_id, obj_meta);

                    NvDsUserMeta *user_event_meta =
                        nvds_acquire_user_meta_from_pool (batch_meta);
                    if (user_event_meta) {
                    user_event_meta->user_meta_data = (void *) msg_meta;
                    user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
                    user_event_meta->base_meta.copy_func =
                        (NvDsMetaCopyFunc) meta_copy_func;
                    user_event_meta->base_meta.release_func =
                        (NvDsMetaReleaseFunc) meta_free_func;
                    nvds_add_user_meta_to_frame (frame_meta, user_event_meta);
                    } else {
                    g_print ("Error in attaching event meta to buffer\n");
                    }
                }
                // is_first_object = FALSE;
            }
        }
    }
    g_print ("Frame Number = %d "
        "Person Count = %d\n",
        frame_number, person_count);
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

    // Return absolute path of config file if file_path is NULL.
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

// Function to read tracker config file and set properties of the GstElement
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

static void
check_gst_element_creation_success(GstElement *element, gchar *component_name)
{
    if (!element) {
        g_printerr("%s element could not be created\n", component_name);
    }
}

int
main (int argc, char *argv[])
{
    // Initialize elements of the DS pipeline
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
        *decoder = NULL, *sink = NULL, *pgie = NULL, *sgie = NULL,
        *nvvidconv = NULL, *nvosd = NULL, *nvstreammux = NULL, *nvtracker = NULL;
    GstElement *msgconv = NULL, *msgbroker = NULL, *tee = NULL;
    GstElement *queue1 = NULL, *queue2 = NULL;
    GstElement *transform = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;
    GstPad *tee_render_pad = NULL;
    GstPad *tee_msg_pad = NULL;
    GstPad *sink_pad = NULL;
    GstPad *src_pad = NULL;
    GOptionContext *ctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;

    int current_device = -1;
    cudaGetDevice (&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    ctx = g_option_context_new("DeepStream Retail IVA");
    group = g_option_group_new("RetailIVA", NULL, NULL, NULL, NULL);
    g_option_group_add_entries(group, entries);

    g_option_context_set_main_group(ctx, group);
    g_option_context_add_group(ctx, gst_init_get_option_group());

    if (!g_option_context_parse(ctx, &argc, &argv, &error)) {
        g_option_context_free (ctx);
        g_printerr("%s", error->message);
        return -1;
    }

    if (!proto_lib || !input_file) {
        if (argc > 1 && !IS_YAML (argv[1])) {
            g_printerr ("missing arguments\n");
            g_printerr ("Usage: %s <yml file>\n", argv[0]);
            g_printerr
                ("Usage: %s -i <H264 filename> -p <Proto adaptor library> --conn-str=<Connection string>\n",
                argv[0]);
            return -1;
        } else if (!argv[1]) {
            g_printerr ("missing arguments\n");
            g_printerr ("Usage: %s <yml file>\n", argv[0]);
            g_printerr
                ("Usage: %s -i <H264 filename> -p <Proto adaptor library> --conn-str=<Connection string>\n",
                argv[0]);
            return -1;
        }
    }

    loop = g_main_loop_new (NULL, FALSE);

    // Create gstreamer elements
    // Create pipeline element that will hold connection of all other elements
    pipeline = gst_pipeline_new ("retail-iva-pipeline");

    // Source element for reading the input file
    source = gst_element_factory_make ("filesrc", "file-source");

    // h264parser to parse input file
    h264parser = gst_element_factory_make("h264parse", "h264-parser");

    // nvdec_h264 for hardware accelerated decoding on GPU
    decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");

    nvstreammux = gst_element_factory_make ("nvstreammux", "nvstreammux");

    // nvinfer to run inferencing on decoder's output - PGIE
    pgie = gst_element_factory_make("nvinfer", "primary-inference-engine");

    // nvinfer to run inferencing on PGIE's output - SGIE
    sgie = gst_element_factory_make("nvinfer", "secondary-inference-engine");

    // tracker to track objects detected by PGIE
    nvtracker = gst_element_factory_make("nvtracker", "tracker");

    // converter to convert from NV12 to RGBA
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

    // osd to draw on converted RGBA buffer
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    // Create message converter to generate payload from buffer metadata
    msgconv = gst_element_factory_make("nvmsgconv", "nvmsg-converter");

    // Create message broker to send payload to kafka server
    msgbroker = gst_element_factory_make("nvmsgbroker", "nvmsg-broker");

    // Create teeo to render buffer and send messages simultaneously
    tee = gst_element_factory_make("tee", "nvsink-tee");

    // Create queues
    queue1 = gst_element_factory_make("queue", "nvtee-que1");
    queue2 = gst_element_factory_make("queue", "nvtee-que2");

    // Finally render the osd output
    if (display_off) {
        sink = gst_element_factory_make("fakesink", "nvvideo-renderer");

    } else {
        sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
        if (prop.integrated) {
            transform = 
                gst_element_factory_make("nvegltransform", "nvegl-transform");
            if (!transform) {
                g_printerr("nvegltransform element could not be created. Exiting\n");
                return -1;
            }
        }
    }


    // Check if the pipeline and all elements are created
    if (!pipeline || !source || !h264parser || !decoder || !nvstreammux || !pgie ||
        !sgie || !nvtracker || !nvvidconv || !nvosd || !msgconv || !msgbroker ||
        !tee || !queue1 || !queue2 || !sink) {
        // Check which element was not created
        check_gst_element_creation_success(pipeline, "pipeline");
        check_gst_element_creation_success(source, "source");
        check_gst_element_creation_success(h264parser, "h264parser");
        check_gst_element_creation_success(decoder, "decoder");
        check_gst_element_creation_success(nvstreammux, "nvstreammux");
        check_gst_element_creation_success(pgie, "pgie");
        check_gst_element_creation_success(sgie, "sgie");
        check_gst_element_creation_success(nvtracker, "nvtracker");
        check_gst_element_creation_success(nvvidconv, "nvvidconv");
        check_gst_element_creation_success(nvosd, "nvosd");
        check_gst_element_creation_success(msgconv, "msgconv");
        check_gst_element_creation_success(msgbroker, "msgbroker");
        check_gst_element_creation_success(tee, "tee");
        check_gst_element_creation_success(queue1, "queue1");
        check_gst_element_creation_success(queue2, "queue2");
        check_gst_element_creation_success(sink, "sink");
        g_printerr("One above element could not be created. Exiting \n");
        return -1;
    }

    if (argc > 1 && IS_YAML (argv[1])) {
        nvds_parse_file_source (source, argv[1], "source");
        nvds_parse_streammux (nvstreammux, argv[1], "streammux");

        g_object_set (G_OBJECT (pgie),
            "config-file-path", "configs/pgie_config_peoplenet.yml", NULL);

        g_object_set (G_OBJECT(sgie),
            "config-file-path", "configs/basket_classifier.yml", NULL);

        g_object_set (G_OBJECT (msgconv), "config", "configs/dstest4_msgconv_config.yml",
            NULL);
        nvds_parse_msgconv (msgconv, argv[1], "msgconv");

        nvds_parse_msgbroker (msgbroker, argv[1], "msgbroker");

        if (display_off)
        nvds_parse_file_sink (sink, argv[1], "sink");
        else
        nvds_parse_egl_sink (sink, argv[1], "sink");

    } else {
        /* we set the input filename to the source element */
        g_object_set (G_OBJECT (source), "location", input_file, NULL);

        g_object_set (G_OBJECT (nvstreammux), "batch-size", 1, NULL);

        g_object_set (G_OBJECT (nvstreammux), "width", MUXER_OUTPUT_WIDTH, "height",
            MUXER_OUTPUT_HEIGHT,
            "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

        /* Set all the necessary properties of the nvinfer element,
        * the necessary ones are : */
        g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);

        g_object_set (G_OBJECT (msgconv), "config", MSCONV_CONFIG_FILE, NULL);
        g_object_set (G_OBJECT (msgconv), "payload-type", schema_type, NULL);
        g_object_set (G_OBJECT (msgconv), "msg2p-newapi", msg2p_meta, NULL);
        g_object_set (G_OBJECT (msgconv), "frame-interval", frame_interval, NULL);

        g_object_set (G_OBJECT (msgbroker), "proto-lib", proto_lib,
            "conn-str", conn_str, "sync", FALSE, NULL);

        if (topic) {
        g_object_set (G_OBJECT (msgbroker), "topic", topic, NULL);
        }

        if (cfg_file) {
        g_object_set (G_OBJECT (msgbroker), "config", cfg_file, NULL);
        }
        gchar *filepath;
        filepath = g_strconcat("output",".mp4",NULL);
        g_object_set (G_OBJECT (sink), "sync", TRUE, NULL);
    }

    if (!set_tracker_properties(nvtracker)) {
        g_printerr ("Failed to set tracker properties. Exiting \n");
        return -1;
    }

    // Adding a message handler
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // Setup a pipeline
    // Add all elements to the pipeline
    gst_bin_add_many(GST_BIN(pipeline),
        source, h264parser, decoder, nvstreammux, pgie, sgie, nvtracker,
        nvvidconv, nvosd, tee, queue1, queue2, msgconv, msgbroker, sink, NULL);

    if (prop.integrated) {
        if (!display_off)
            gst_bin_add (GST_BIN(pipeline), transform);
    }

    // Link elements together
    /*
    file-source -> h264-parser -> nvh264-decoder -> nvstreammux ->
    nvinfer -> nvtracker -> nvvidconv -> nvosd -> tee -> video-renderer
                                            |
                                            | -> msgconv -> msgbroker
    */
    sink_pad = gst_element_get_request_pad(nvstreammux, "sink_0");
    if (!sink_pad) {
        g_printerr("Streammux request sink pad failed. Exiting \n");
        return -1;
    }

    src_pad = gst_element_get_static_pad(decoder, "src");
    if (!src_pad) {
        g_printerr("Decoder request src pad failed. Exiting \n");
        return -1;
    }

    if (gst_pad_link(src_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link decoder to stream muxer. Exiting \n");
        return -1;
    }

    gst_object_unref(sink_pad);
    gst_object_unref(src_pad);
    
    if (!gst_element_link_many(source, h264parser, decoder, NULL)) {
        g_printerr("Elements could not be linked. Exiting \n");
        return -1;
    }

    if (!gst_element_link_many(nvstreammux, pgie, nvtracker, sgie, nvvidconv, nvosd, tee, NULL)) {
        g_printerr ("Elements could not be linked. Exiting \n");
        return -1;
    }

    if (!gst_element_link_many(queue1, msgconv, msgbroker, NULL)) {
        g_printerr("Elements could not be linked. Exiting \n");
        return -1;
    }

    if (prop.integrated) {
        if (!display_off) {
            if(!gst_element_link_many(queue2, transform, sink, NULL)) {
                g_printerr("Elements could not be linked. Exiting \n");
                return -1;
            }
        } else {
            if (!gst_element_link(queue2, sink)) {
                g_printerr("Elements could not be linked. Exiting \n");
                return -1;
            }
        } 
    } else {
        if (!gst_element_link (queue2, sink)) {
            g_printerr ("Elements could not be linked. Exiting.\n");
            return -1;
        }
    }

    sink_pad = gst_element_get_static_pad (queue1, "sink");
    tee_msg_pad = gst_element_get_request_pad(tee, "src_%u");
    tee_render_pad = gst_element_get_request_pad(tee, "src_%u");

    if (!tee_msg_pad || !tee_render_pad) {
        g_printerr("Unable to request pads \n");
        return -1;
    }

    if (gst_pad_link(tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Unable to link tee and message converter. \n");
        gst_object_unref(sink_pad);
        return -1;
    }

    gst_object_unref(sink_pad);

    sink_pad = gst_element_get_static_pad(queue2, "sink");
    if (gst_pad_link(tee_render_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Unable to link tee and render pad. \n");
        gst_object_unref (sink_pad);
        return -1;
    }

    gst_object_unref(sink_pad);

    // Adding a probe to get informed of the meta data generated.
    // We add a probe to the sink pad of the OSD element since by 
    // that time the buffer would have had got all the metadata

    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad) {
        g_print("Unable to get sink pad\n");
    } else {
        if (msg2p_meta == 0) {
            gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, 
                osd_sink_pad_buffer_probe, NULL, NULL);
        }
    }
    gst_object_unref(osd_sink_pad);

    // Set pipeline to playing state
    if (argc > 1 && IS_YAML(argv[1])) {
        g_print ("Using file %s \n", argv[1]);
    } else {
        g_print("Now playing %s\n", input_file);
    }
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Wait till the pipeline encounters an error or End of Stream (EOS)
    g_print("Running ...\n");
    g_main_loop_run(loop);

    // Out of the main loop. Perform clean up
    g_print ("Returned, stopped playback\n");

    g_free (cfg_file);
    g_free(input_file);
    g_free(topic);
    g_free(conn_str);
    g_free(proto_lib);

    // Release the request pads from tee and unfer them
    gst_element_release_request_pad(tee, tee_msg_pad);
    gst_element_release_request_pad(tee, tee_render_pad);
    gst_object_unref(tee_msg_pad);
    gst_object_unref(tee_render_pad);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline \n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
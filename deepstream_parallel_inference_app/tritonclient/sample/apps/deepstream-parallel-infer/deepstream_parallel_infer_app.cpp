/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "deepstream_parallel_infer.h"
#include "post_process/body_pose/post_process.cpp"
#include "nvds_version.h"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"
#include "nvdsmeta_schema.h"
#include "deepstream_perf.h"

#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>

#define EPS 1e-6

#define MAX_DISPLAY_LEN 64

#define MAX_TIME_STAMP_LEN 32

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MAX_STR_LEN 2048

AppCtx *appCtx;
static guint cintr = FALSE;
static GMainLoop *main_loop = NULL;
static gchar **cfg_files = NULL;
static gboolean print_version = FALSE;
static gboolean show_bbox_text = FALSE;
static gboolean print_dependencies_version = FALSE;
static gboolean quit = FALSE;
static gint return_value = 0;
static guint num_input_uris;
static gint frame_interval = 30;

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

gint frame_number = 0;

#define MAX_STREAMS 64

typedef struct
{
    /** identifies the stream ID */
    guint32 stream_index;
    gdouble fps[MAX_STREAMS];
    gdouble fps_avg[MAX_STREAMS];
    guint32 num_instances;
    guint header_print_cnt;
    GMutex fps_lock;
    gpointer context;

    /** Test specific info */
    guint32 set_batch_size;
}DemoPerfCtx;

typedef struct {
  GMutex *lock;
  int num_sources;
}LatencyCtx;

/**
 * callback function to print the performance numbers of each stream.
 */
static void
perf_cb (gpointer context, NvDsAppPerfStruct * str)
{
  DemoPerfCtx *thCtx = (DemoPerfCtx *) context;

  g_mutex_lock(&thCtx->fps_lock);
  /** str->num_instances is == num_sources */
  guint32 numf = str->num_instances;
  guint32 i;

  for (i = 0; i < numf; i++) {
    thCtx->fps[i] = str->fps[i];
    thCtx->fps_avg[i] = str->fps_avg[i];
  }
  thCtx->context = thCtx;
  g_print ("**PERF: ");
  for (i = 0; i < numf; i++) {
    g_print ("%.2f (%.2f)\t", thCtx->fps[i], thCtx->fps_avg[i]);
  }
  g_print ("\n");
  g_mutex_unlock(&thCtx->fps_lock);
}

/**
 * callback function to print the latency of each component in the pipeline.
 */

static GstPadProbeReturn
latency_measurement_buf_prob(GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  LatencyCtx *ctx = (LatencyCtx *) u_data;
  static int batch_num = 0;
  guint i = 0, num_sources_in_batch = 0;
  if(nvds_enable_latency_measurement)
  {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameLatencyInfo *latency_info = NULL;
    g_mutex_lock (ctx->lock);
    latency_info = (NvDsFrameLatencyInfo *)
      calloc(1, ctx->num_sources * sizeof(NvDsFrameLatencyInfo));;
    g_print("\n************BATCH-NUM = %d soure %d**************\n",batch_num,ctx->num_sources);
    num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

    for(i = 0; i < num_sources_in_batch; i++)
    {
      g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
          latency_info[i].source_id,
          latency_info[i].frame_num,
          latency_info[i].latency);
    }
    g_mutex_unlock (ctx->lock);
    batch_num++;
  }

  return GST_PAD_PROBE_OK;
}

GST_DEBUG_CATEGORY (NVDS_APP);

GOptionEntry entries[] = {
  {"version", 'v', 0, G_OPTION_ARG_NONE, &print_version,
      "Print DeepStreamSDK version", NULL}
  ,
  {"tiledtext", 't', 0, G_OPTION_ARG_NONE, &show_bbox_text,
      "Display Bounding box labels in tiled mode", NULL}
  ,
  {"version-all", 0, 0, G_OPTION_ARG_NONE, &print_dependencies_version,
      "Print DeepStreamSDK and dependencies version", NULL}
  ,
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files,
      "Set the config file", NULL}
  ,
  {NULL}
  ,
};

/*Method to parse information returned from the model*/
std::tuple<Vec2D<int>, Vec3D<float>>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
  Vec1D<int> counts;
  Vec3D<int> peaks;

  float threshold = 0.1;
  int window_size = 5;
  int max_num_parts = 20;
  int num_integral_samples = 7;
  float link_threshold = 0.1;
  int max_num_objects = 100;

  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  void *paf_data = tensor_meta->out_buf_ptrs_host[1];
  NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;

  /* Finding peaks within a given window */
  find_peaks(counts, peaks, cmap_data, cmap_dims, threshold, window_size, max_num_parts);
  /* Non-Maximum Suppression */
  Vec3D<float> refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size);
  /* Create a Bipartite graph to assign detected body-parts to a unique person in the frame */
  Vec3D<float> score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples);
  /* Assign weights to all edges in the bipartite graph generated */
  Vec3D<int> connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts);
  /* Connecting all the Body Parts and Forming a Human Skeleton */
  Vec2D<int> objects = connect_parts(connections, topology, counts, max_num_objects);
  return {objects, refined_peaks};
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
  int K = topology.size();
  int count = objects.size();
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  for (auto &object : objects)
  {
    int C = object.size();
    for (int j = 0; j < C; j++)
    {
      int k = object[j];
      if (k >= 0)
      {
        auto &peak = normalized_peaks[j][k];
        int x = peak[1] * MUXER_OUTPUT_WIDTH;
        int y = peak[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
        cparams.xc = x;
        cparams.yc = y;
        cparams.radius = 8;
        cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
        cparams.has_bg_color = 1;
        cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_circles++;
      }
    }

    for (int k = 0; k < K; k++)
    {
      int c_a = topology[k][2];
      int c_b = topology[k][3];
      if (object[c_a] >= 0 && object[c_b] >= 0)
      {
        auto &peak0 = normalized_peaks[c_a][object[c_a]];
        auto &peak1 = normalized_peaks[c_b][object[c_b]];
        int x0 = peak0[1] * MUXER_OUTPUT_WIDTH;
        int y0 = peak0[0] * MUXER_OUTPUT_HEIGHT;
        int x1 = peak1[1] * MUXER_OUTPUT_WIDTH;
        int y1 = peak1[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
        lparams.x1 = x0;
        lparams.x2 = x1;
        lparams.y1 = y0;
        lparams.y2 = y1;
        lparams.line_width = 3;
        lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_lines++;
      }
    }
  }
}

/* body_pose_gie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
body_pose_gie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    if (frame_meta->batch_id == 0)
      g_print("Processing frame number = %d\t\n", frame_meta->frame_num);

    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
         l_user = l_user->next)
    {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
      {
        NvDsInferTensorMeta *tensor_meta =
            (NvDsInferTensorMeta *)user_meta->user_meta_data;
        Vec2D<int> objects;
        Vec3D<float> normalized_peaks;
        tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
        create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
      }
    }

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
          Vec2D<int> objects;
          Vec3D<float> normalized_peaks;
          tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
          create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
yolov4_gie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
         l_user = l_user->next)
    {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
      {
        NvDsInferTensorMeta *tensor_meta =
            (NvDsInferTensorMeta *)user_meta->user_meta_data;
      }
    }

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

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

static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *) g_memdup (srcMeta, sizeof (NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->sensorStr)
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature =  (gdouble *) g_memdup (srcMeta->objSignature.signature,
        srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *srcObj = (NvDsVehicleObject *) srcMeta->extMsg;
      NvDsVehicleObject *obj =
          (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->make)
        obj->make = g_strdup (srcObj->make);
      if (srcObj->model)
        obj->model = g_strdup (srcObj->model);
      if (srcObj->color)
        obj->color = g_strdup (srcObj->color);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->region)
        obj->region = g_strdup (srcObj->region);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsVehicleObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
      NvDsPersonObject *obj =
          (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

      obj->age = srcObj->age;

      if (srcObj->gender)
        obj->gender = g_strdup (srcObj->gender);
      if (srcObj->cap)
        obj->cap = g_strdup (srcObj->cap);
      if (srcObj->hair)
        obj->hair = g_strdup (srcObj->hair);
      if (srcObj->apparel)
        obj->apparel = g_strdup (srcObj->apparel);
      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsPersonObject);
    }
  }

  return dstMeta;
}

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
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->color)
        g_free (obj->color);
      if (obj->make)
        g_free (obj->make);
      if (obj->model)
        g_free (obj->model);
      if (obj->license)
        g_free (obj->license);
      if (obj->region)
        g_free (obj->region);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

      if (obj->gender)
        g_free (obj->gender);
      if (obj->cap)
        g_free (obj->cap);
      if (obj->hair)
        g_free (obj->hair);
      if (obj->apparel)
        g_free (obj->apparel);
    }
    g_free (srcMeta->extMsg);
    srcMeta->extMsgSize = 0;
  }
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

static void
generate_vehicle_meta (gpointer data)
{
  NvDsVehicleObject *obj = (NvDsVehicleObject *) data;

  obj->type = g_strdup ("sedan");
  obj->color = g_strdup ("blue");
  obj->make = g_strdup ("Bugatti");
  obj->model = g_strdup ("M");
  obj->license = g_strdup ("XX1234");
  obj->region = g_strdup ("CA");
}

static void
generate_person_meta (gpointer data)
{
  NvDsPersonObject *obj = (NvDsPersonObject *) data;
  obj->age = 45;
  obj->cap = g_strdup ("none");
  obj->hair = g_strdup ("black");
  obj->gender = g_strdup ("male");
  obj->apparel = g_strdup ("formal");
}

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

  /*
   * This demonstrates how to attach custom objects.
   * Any custom object as per requirement can be generated and attached
   * like NvDsVehicleObject / NvDsPersonObject. Then that object should
   * be handled in payload generator library (nvmsgconv.cpp) accordingly.
   */
  if (class_id == PGIE_CLASS_ID_VEHICLE) {
    meta->type = NVDS_EVENT_MOVING;
    meta->objType = NVDS_OBJECT_TYPE_VEHICLE;
    meta->objClassId = PGIE_CLASS_ID_VEHICLE;

    NvDsVehicleObject *obj =
        (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
    generate_vehicle_meta (obj);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsVehicleObject);
  } else if (class_id == PGIE_CLASS_ID_PERSON) {
    meta->type = NVDS_EVENT_ENTRY;
    meta->objType = NVDS_OBJECT_TYPE_PERSON;
    meta->objClassId = PGIE_CLASS_ID_PERSON;

    NvDsPersonObject *obj =
        (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
    generate_person_meta (obj);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsPersonObject);
  }
}
/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;
  gboolean is_first_object = TRUE;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    is_first_object = TRUE;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
      /** Generate NvDsEventMsgMeta for every object */
      if (is_first_object && !(frame_number % frame_interval)) {
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
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number =  %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    txt_params->font_params.font_name = "Mono";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *main_loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(main_loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(main_loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

gboolean
link_element_to_metamux_sink_pad (GstElement *metamux, GstElement *elem,
    gint index)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;
  gchar pad_name[16];

  if (index >= 0) {
    g_snprintf (pad_name, 16, "sink_%u", index);
    pad_name[15] = '\0';
  } else {
    strcpy (pad_name, "sink_%u");
  }

  mux_sink_pad = gst_element_get_request_pad (metamux, pad_name);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from metamux");
    goto done;
  }

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
                        GST_ELEMENT_NAME (elem));
    goto done;
  }

  if (gst_pad_link (src_pad, mux_sink_pad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (metamux),
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

gboolean
unlink_element_from_metamux_sink_pad (GstElement *metamux, GstElement *elem)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
                        GST_ELEMENT_NAME (elem));
    goto done;
  }

  mux_sink_pad = gst_pad_get_peer (src_pad);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from metamux");
    goto done;
  }

  if (!gst_pad_unlink (src_pad, mux_sink_pad)) {
    NVGSTDS_ERR_MSG_V ("Failed to unlink '%s' and '%s'", GST_ELEMENT_NAME (metamux),
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  gst_element_release_request_pad(metamux, mux_sink_pad);

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

gboolean
link_streamdemux_to_streammux (NvDsParallelGieBin *bin, GstElement *demux, GstElement *mux,
    gint index)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *source_tee_src_pad = NULL;
  GstElement *queue = NULL;
  gchar pad_name[16];

  if (!bin->source_tee[index]) {
    bin->source_tee[index] = gst_element_factory_make (NVDS_ELEM_TEE, NULL);
    if (!bin->source_tee[index]) {
      NVGSTDS_ERR_MSG_V ("Failed to create 'infer_bin_source_tee'");
      goto done;
    }
    gst_bin_add (GST_BIN (bin->bin), bin->source_tee[index]);

    link_element_to_demux_src_pad (demux, bin->source_tee[index], index);
  }

  queue = gst_element_factory_make (NVDS_ELEM_QUEUE, NULL);
  if (!queue) {
    NVGSTDS_ERR_MSG_V ("Could not create 'queue'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), queue);
  link_element_to_streammux_sink_pad (mux, queue, index);

  link_element_to_tee_src_pad (bin->source_tee[index], queue);

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (source_tee_src_pad) {
    gst_object_unref (source_tee_src_pad);
  }
  return ret;
}

gboolean
create_primary_gie_videotemplate_bin (NvDsVideoTemplateConfig *config, NvDsPrimaryGieBin *bin)
{
  gboolean ret = FALSE;
  guint i;

  bin->bin = gst_bin_new ("primary_gie_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'primary_gie_bin'");
    goto done;
  }

  bin->nvvidconv =
      gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "primary_gie_conv");
  if (!bin->nvvidconv) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'primary_gie_conv'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "primary_gie_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'primary_gie_queue'");
    goto done;
  }

  bin->primary_gie =
       gst_element_factory_make ("nvdsvideotemplate", "primary_gie");
  if (!bin->primary_gie) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'primary_gie'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->primary_gie),
    "customlib-name", config->customlib_name, NULL);

  for (i = 0; i < config->num_customlib_props; i ++) {
    g_object_set (G_OBJECT (bin->primary_gie),
      "customlib-props", config->customlib_props[i], NULL);
  }

  /*
  g_object_set (G_OBJECT (bin->nvvidconv), "gpu-id", config->gpu_id, NULL);
  g_object_set (G_OBJECT (bin->nvvidconv), "nvbuf-memory-type",
      config->nvbuf_memory_type, NULL);
  */

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue,
      bin->nvvidconv, bin->primary_gie, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->nvvidconv);

  NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->primary_gie);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->primary_gie, "src");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

#if 0
static gboolean
create_parallel_infer_bin (guint num_sub_bins, NvDsConfig *config,
    NvDsParallelGieBin *bin, AppCtx *appCtx)
{
  gboolean ret = FALSE;
  GstElement *sink_elem = NULL;
  GstElement *src_elem = NULL;
  GstElement *nvvidconv = NULL, *caps_filter = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  gchar name[50];
  guint i = 0;

  bin->bin = gst_bin_new ("parallel_infer_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'parallel_infer_bin'");
    goto done;
  }

  bin->tee = gst_element_factory_make (NVDS_ELEM_TEE, "infer_bin_tee");
  if (!bin->tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'infer_bin_tee'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), bin->tee);

  bin->muxer = gst_element_factory_make ("nvdsmetamux", "infer_bin_muxer");
  if (!bin->muxer) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'infer_bin_muxer'");
    goto done;
  }
  g_object_set (G_OBJECT (bin->muxer), "config-file",
		 GET_FILE_PATH (config->meta_mux_config.config_file_path), NULL);

  NVGSTDS_ELEM_ADD_PROBE (bin->muxer_buffer_probe_id, bin->muxer, "src",
      body_pose_gie_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER,
      appCtx);

  gst_bin_add (GST_BIN (bin->bin), bin->muxer);

  for (i = 0; i < num_sub_bins; i++) {
    if (config->primary_gie_sub_bin_config[i].enable
		    || config->video_template_sub_bin_config[i].enable) {
      if (config->video_template_sub_bin_config[i].enable) {
        if (!create_primary_gie_videotemplate_bin (&config->video_template_sub_bin_config[i],
              &bin->primary_gie_bin[i])) {
          goto done;
        }
      } else {
        if (!create_primary_gie_bin (&config->primary_gie_sub_bin_config[i],
              &bin->primary_gie_bin[i])) {
          goto done;
        }
      }
      g_snprintf (name, sizeof (name), "primary_gie_%d_bin", i);
      gst_element_set_name (bin->primary_gie_bin[i].bin, name);
      gst_bin_add (GST_BIN (bin->bin), bin->primary_gie_bin[i].bin);

      sink_elem = bin->primary_gie_bin[i].bin;
      src_elem = bin->primary_gie_bin[i].bin;
    }

    if (config->pre_process_sub_bin_config[i].enable) {
      if (!create_preprocess_bin (&config->pre_process_sub_bin_config[i],
            &bin->preprocess_bin[i])) {
        g_print ("creating preprocess bin failed\n");
        goto done;
      }
      g_snprintf (name, sizeof (name), "preprocess_%d_bin", i);
      gst_element_set_name (bin->preprocess_bin[i].bin, name);
      gst_bin_add (GST_BIN (bin->bin), bin->preprocess_bin[i].bin);

      if (sink_elem) {
        NVGSTDS_LINK_ELEMENT (bin->preprocess_bin[i].bin, sink_elem);
      }

      sink_elem = bin->preprocess_bin[i].bin;
    }

    /* Add video convert to avoid parallel infer operate on the same batch meta */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", NULL);
    caps_filter = gst_element_factory_make ("capsfilter", NULL);
    caps =
        gst_caps_new_simple ("video/x-raw",
        "width", G_TYPE_INT, 1920,
        "height", G_TYPE_INT, 1082,
        NULL);
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);
    g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);
    gst_bin_add (GST_BIN (bin->bin), nvvidconv);
    gst_bin_add (GST_BIN (bin->bin), caps_filter);
    NVGSTDS_LINK_ELEMENT (nvvidconv, caps_filter);
    NVGSTDS_LINK_ELEMENT (caps_filter, sink_elem);
    sink_elem = nvvidconv;

    link_element_to_tee_src_pad (bin->tee, sink_elem);
    link_element_to_metamux_sink_pad (bin->muxer, src_elem, i);
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->tee, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->muxer, "src");

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
#endif

/* Separate a config file entry with delimiters
 * into strings. */
std::vector<std::string> split_string (std::string input) {
  std::vector<int> positions;
  for (unsigned int i = 0; i < input.size(); i++) {
    if (input[i] == ';')
      positions.push_back(i);
  }
  std::vector<std::string> ret;
  int prev = 0;
  for (auto &j: positions) {
    std::string temp = input.substr(prev, j - prev);
    ret.push_back(temp);
    prev = j + 1;
  }
  ret.push_back(input.substr(prev, input.size() - prev));
  return ret;
}

/* select streamdemux output sources, then link sources and streammux */
static gboolean link_streamdemux_to_streammux(NvDsConfig *config, NvDsParallelGieBin *bin, int i){
    gboolean ret = FALSE;
    if (config->primary_gie_sub_bin_config[i].unique_id != config->srcids_config[i].pgie_id) {
        NVGSTDS_ERR_MSG_V ("pgieid %d != branch pgieid %d\n",
        config->primary_gie_sub_bin_config[i].unique_id,
         config->srcids_config[i].pgie_id);
         return ret;
    } else {
      std::string str = config->srcids_config[i].src_ids;
      std::vector<std::string> vec = split_string (str);
      for(int j = 0; j < vec.size(); j++) {
        int id = std::stoi(vec[j]);
	    g_print("link_streamdemux_to_streammux, srid:%d, mux:%d\n", id, i);
        if (!link_streamdemux_to_streammux (bin, bin->demuxer, bin->streammux[i], id)) {
          NVGSTDS_ERR_MSG_V ("source %d cannot be linked to mux's sink pad %p\n",
            id, bin->streammux[i]);
          return ret;
        }
      }
    }
    return true;
}

static gboolean
create_parallel_infer_bin (guint num_sub_bins, NvDsConfig *config,
    NvDsParallelGieBin *bin, AppCtx *appCtx)
{
  gboolean ret = FALSE;
  GstElement *sink_elem = NULL;
  GstElement *src_elem = NULL;
  GstElement *queue = NULL;
  GstElement *nvvidconv = NULL, *caps_filter = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  gchar name[50];
  guint i, j;
  std::string str;
  std::vector<std::string> vec;
  guint src_id_num;

  bin->bin = gst_bin_new ("parallel_infer_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'parallel_infer_bin'");
    goto done;
  }

  bin->tee = gst_element_factory_make (NVDS_ELEM_TEE, "infer_bin_tee");
  if (!bin->tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'infer_bin_tee'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), bin->tee);

  bin->muxer = gst_element_factory_make ("nvdsmetamux", "infer_bin_muxer");
  if (!bin->muxer) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'infer_bin_muxer'");
    goto done;
  }
  g_object_set (G_OBJECT (bin->muxer), "config-file",
		 GET_FILE_PATH (config->meta_mux_config.config_file_path), NULL);

  NVGSTDS_ELEM_ADD_PROBE (bin->muxer_buffer_probe_id, bin->muxer, "src",
      body_pose_gie_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER,
      appCtx);

  gst_bin_add (GST_BIN (bin->bin), bin->muxer);
  sink_elem = bin->muxer;

  queue = gst_element_factory_make (NVDS_ELEM_QUEUE, NULL);
  if (!queue) {
    NVGSTDS_ERR_MSG_V ("Could not create 'queue'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), queue);
  link_element_to_metamux_sink_pad (bin->muxer, queue, 0);
  sink_elem = queue;

  link_element_to_tee_src_pad (bin->tee, sink_elem);

  bin->demuxer =
      gst_element_factory_make (NVDS_ELEM_STREAM_DEMUX, NULL);
  if (!bin->demuxer) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'demuxer'");
    goto done;
  }
  g_object_set (G_OBJECT (bin->demuxer), "per-stream-eos", TRUE, NULL);
  gst_bin_add (GST_BIN (bin->bin), bin->demuxer);
  sink_elem = bin->demuxer;

  queue = gst_element_factory_make (NVDS_ELEM_QUEUE, NULL);
  if (!queue) {
    NVGSTDS_ERR_MSG_V ("Could not create 'queue'");
    goto done;
  }
  gst_bin_add (GST_BIN (bin->bin), queue);
  NVGSTDS_LINK_ELEMENT (queue, sink_elem);
  sink_elem = queue;

  link_element_to_tee_src_pad (bin->tee, sink_elem);


  for (i = 0; i < num_sub_bins; i++) {
    sink_elem = src_elem = NULL;

    if (config->primary_gie_sub_bin_config[i].enable
                    || config->video_template_sub_bin_config[i].enable) {

      if (config->num_secondary_gie_sub_bins > 0 && config->num_secondary_gie_num[i] > 0) {
         if (!create_secondary_gie_bin (config->num_secondary_gie_num[i],
              config->primary_gie_sub_bin_config[i].unique_id,
              config->secondary_gie_sub_bin_config[i],
              &bin->secondary_gie_bin[i])) {
           g_print("create_secondary_gie_bin failed");
           goto done;
         }
         g_snprintf (name, sizeof (name), "sgie_%d_bin", i);
         gst_element_set_name (bin->secondary_gie_bin[i].bin, name);
      	 gst_bin_add (GST_BIN (bin->bin), bin->secondary_gie_bin[i].bin);
         sink_elem = bin->secondary_gie_bin[i].bin;
         src_elem = bin->secondary_gie_bin[i].bin;
       }  
    }
    //add analysis
    if (config->tracker_config[i].enable && config->dsanalytics_config[i].enable) {
      if (!create_dsanalytics_bin (&config->dsanalytics_config[i],
              &bin->dsanalytics_bin[i])) {
        g_print ("creating dsanalytics bin failed\n");
        goto done;
      }
      
      g_snprintf (name, sizeof (name), "analytics_%d_bin", i);
      gst_element_set_name (bin->dsanalytics_bin[i].bin, name);
      gst_bin_add (GST_BIN (bin->bin), bin->dsanalytics_bin[i].bin);
      if (sink_elem) {
        NVGSTDS_LINK_ELEMENT (bin->dsanalytics_bin[i].bin, sink_elem);
      }
      sink_elem = bin->dsanalytics_bin[i].bin;
      if (!src_elem) {
          src_elem = bin->dsanalytics_bin[i].bin;
      }
    }
    //add tracker
    if (config->tracker_config[i].enable) {
      if (!create_tracking_bin (&config->tracker_config[i],
              &bin->tracker_bin[i])) {
        g_print ("creating tracker bin failed\n");
        goto done;
      }
      
      g_snprintf (name, sizeof (name), "tracking_%d_bin", i);
      gst_element_set_name (bin->tracker_bin[i].bin, name);
      gst_bin_add (GST_BIN (bin->bin),
          bin->tracker_bin[i].bin);

      if (sink_elem) {
        NVGSTDS_LINK_ELEMENT (bin->tracker_bin[i].bin, sink_elem);
      }
      sink_elem = bin->tracker_bin[i].bin;     
      if (!src_elem) {
         src_elem = bin->tracker_bin[i].bin;
      }
    }
	  
    if (config->primary_gie_sub_bin_config[i].enable
		    || config->video_template_sub_bin_config[i].enable) {
      if (config->video_template_sub_bin_config[i].enable) {
        if (!create_primary_gie_videotemplate_bin (&config->video_template_sub_bin_config[i],
              &bin->primary_gie_bin[i])) {
          goto done;
        }
      } else {
        if (!create_primary_gie_bin (&config->primary_gie_sub_bin_config[i],
              &bin->primary_gie_bin[i])) {
          goto done;
        }
      }
      g_snprintf (name, sizeof (name), "primary_gie_%d_bin", i);
      gst_element_set_name (bin->primary_gie_bin[i].bin, name);
      gst_bin_add (GST_BIN (bin->bin), bin->primary_gie_bin[i].bin);

      if (sink_elem) {
        NVGSTDS_LINK_ELEMENT (bin->primary_gie_bin[i].bin, sink_elem);
      }
     
      sink_elem = bin->primary_gie_bin[i].bin;
      if (!src_elem) {
         src_elem = bin->primary_gie_bin[i].bin;
      }
    }

    if (config->pre_process_sub_bin_config[i].enable) {
      if (!create_preprocess_bin (&config->pre_process_sub_bin_config[i],
            &bin->preprocess_bin[i])) {
        g_print ("creating preprocess bin failed\n");
        goto done;
      }
      g_snprintf (name, sizeof (name), "preprocess_%d_bin", i);
      gst_element_set_name (bin->preprocess_bin[i].bin, name);
      gst_bin_add (GST_BIN (bin->bin), bin->preprocess_bin[i].bin);

      if (sink_elem) {
        NVGSTDS_LINK_ELEMENT (bin->preprocess_bin[i].bin, sink_elem);
      }

      sink_elem = bin->preprocess_bin[i].bin;
    }

    /* streamdemux and streammux to select source to inference */
    bin->streammux[i] =
        gst_element_factory_make (NVDS_ELEM_STREAM_MUX, NULL);
    if (!bin->streammux[i]) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'streammux'");
      goto done;
    }
    gst_bin_add (GST_BIN (bin->bin), bin->streammux[i]);
     if (config->streammux_config.is_parsed){
      if(!set_streammux_properties (&config->streammux_config,
          bin->streammux[i])){
           NVGSTDS_WARN_MSG_V("Failed to set streammux properties");
      }
    }

    str = config->srcids_config[i].src_ids;
    vec = split_string (str);
    src_id_num = vec.size();
    g_print("i:%d, src_id_num:%d\n", i, src_id_num);
    g_object_set (G_OBJECT (bin->streammux[i]), "batch-size", src_id_num, NULL);

    if(!link_streamdemux_to_streammux(config, bin, i)){
        goto done;
    }
    
    NVGSTDS_LINK_ELEMENT (bin->streammux[i], sink_elem);
    
    link_element_to_metamux_sink_pad (bin->muxer, src_elem, i+1);
  }

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->tee, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->muxer, "src");

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

static gboolean
add_and_link_broker_sink (AppCtx * appCtx)
{
  NvDsConfig *config = &appCtx->config;
  /** Only first instance_bin broker sink
   * employed as there's only one analytics path for N sources
   * NOTE: There shall be only one [sink] group
   * with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
   * a) Multiple of them does not make sense as we have only
   * one analytics pipe generating the data for broker sink
   * b) If Multiple broker sinks are configured by the user
   * in config file, only the first in the order of
   * appearance will be considered
   * and others shall be ignored
   * c) Ideally it should be documented (or obvious) that:
   * multiple [sink] groups with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
   * is invalid
   */
  NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[0];
  NvDsPipeline *pipeline = &appCtx->pipeline;

  for (guint i = 0; i < config->num_sink_sub_bins; i++) {
    if(config->sink_bin_sub_bin_config[i].type == NV_DS_SINK_MSG_CONV_BROKER)
    {
      /** add the broker sink bin to pipeline */
      if(!gst_bin_add (GST_BIN (pipeline->pipeline), instance_bin->sink_bin.sub_bins[i].bin)) {
        return FALSE;
      }
      g_print("add_and_link_broker_sink\n");
      // link the broker sink bin to the sink tee
      if (!link_element_to_tee_src_pad (instance_bin->sink_tee, instance_bin->sink_bin.sub_bins[i].bin)) {
        return FALSE;
      }
    }
  }
  return TRUE;
}

int main(int argc, char *argv[])
{
  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GstElement *last_elem = NULL;
  NvDsInstanceBin *instance_bin;
  NvDsPipeline *pipeline;
  NvDsConfig *config;
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL;
  GError *error = NULL;
  guint i;
  const gchar *new_mux_str = NULL;
  gboolean use_new_mux = FALSE;

  ctx = g_option_context_new ("Nvidia DeepStream Parallel Demo");
  group = g_option_group_new ("abc", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (ctx, group);
  g_option_context_add_group (ctx, gst_init_get_option_group ());

  GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

  if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    return -1;
  }

  if (print_version) {
    g_print ("deepstream-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    nvds_version_print ();
    return 0;
  }

  if (print_dependencies_version) {
    g_print ("deepstream-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    nvds_version_print ();
    nvds_dependencies_version_print ();
    return 0;
  }

  if (!cfg_files) {
    NVGSTDS_ERR_MSG_V ("Specify config file with -c option");
    return_value = -1;
    goto done;
  }

  appCtx = (AppCtx *)g_malloc0 (sizeof (AppCtx));
  appCtx->person_class_id = -1;
  appCtx->car_class_id = -1;
  appCtx->index = i;
  appCtx->active_source_index = -1;
  if (show_bbox_text) {
    appCtx->show_bbox_text = TRUE;
  }

  if (!parse_config_file_yaml (&appCtx->config, cfg_files[0])) {
    NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[0]);
    appCtx->return_value = -1;
    goto done;
  }

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  main_loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  instance_bin = &appCtx->pipeline.instance_bins[0];
  pipeline = &appCtx->pipeline;
  config = &appCtx->config;

  pipeline->pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

  /*
   * Add muxer and < N > source components to the pipeline based
   * on the settings in configuration file.
   */
  if (!create_multi_source_bin (config->num_source_sub_bins,
          config->multi_source_config, &pipeline->multi_src_bin)) {
      g_print ("creating multi source bin failed\n");
      goto done;
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->multi_src_bin.bin);

  /* if using new sreammux, nvvideocovnert will scale input resolutions to the same resolution */
  new_mux_str = g_getenv ("USE_NEW_NVSTREAMMUX");
  use_new_mux = !g_strcmp0 (new_mux_str, "yes");
  if (use_new_mux) {
      GstCaps* caps = NULL;
      gchar * caps_string = NULL;
      GstCaps* fiter_caps = NULL;
      char strCaps[MAX_STR_LEN] = {0};
      for (int i = 0; i < config->num_source_sub_bins; i++){
          g_object_get (G_OBJECT (pipeline->multi_src_bin.sub_bins[i].cap_filter1), "caps", &caps, NULL);
          caps_string = gst_caps_to_string (caps);
          snprintf(strCaps, MAX_STR_LEN-1, "%s,width=%d, height=%d", caps_string,
            config->streammux_config.pipeline_width, config->streammux_config.pipeline_height);
          fiter_caps = gst_caps_from_string (strCaps);
          g_object_set (G_OBJECT (pipeline->multi_src_bin.sub_bins[i].cap_filter1), "caps", fiter_caps, NULL);
          printf("strCaps:%s\n", strCaps);

          gst_caps_unref (caps);
          gst_caps_unref (fiter_caps);
          g_free (caps_string);
      }
  }

  if (config->streammux_config.is_parsed){
    if(!set_streammux_properties (&config->streammux_config,
        pipeline->multi_src_bin.streammux)){
         NVGSTDS_WARN_MSG_V("Failed to set streammux properties");
    }
  }

  if (!create_parallel_infer_bin (config->num_primary_gie_sub_bins,
          config, &pipeline->parallel_infer_bin, appCtx)) {
      g_print ("creating parallel infer bin failed\n");
      goto done;
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->parallel_infer_bin.bin);
  last_elem = pipeline->parallel_infer_bin.bin;
  NVGSTDS_LINK_ELEMENT (pipeline->multi_src_bin.bin, last_elem);

  /* Add common message converter */
  if (config->msg_conv_config.enable) {
    NvDsSinkMsgConvBrokerConfig *convConfig = &config->msg_conv_config;
    instance_bin->msg_conv = gst_element_factory_make (NVDS_ELEM_MSG_CONV, "common_msg_conv");
    if (!instance_bin->msg_conv) {
      NVGSTDS_ERR_MSG_V ("Failed to create element 'common_msg_conv'");
      goto done;
    }
    
    g_object_set (G_OBJECT(instance_bin->msg_conv),
          "config", convConfig->config_file_path,
          "msg2p-lib", (convConfig->conv_msg2p_lib ? convConfig->conv_msg2p_lib : "null"),
          "payload-type", convConfig->conv_payload_type,
          "comp-id", convConfig->conv_comp_id,
          "debug-payload-dir", convConfig->debug_payload_dir,
          "multiple-payloads", convConfig->multiple_payloads,
          "msg2p-newapi", convConfig->conv_msg2p_new_api,
          "frame-interval", convConfig->conv_frame_interval,
          NULL);

    gst_bin_add (GST_BIN (pipeline->pipeline),
                  instance_bin->msg_conv);

    NVGSTDS_LINK_ELEMENT (last_elem, instance_bin->msg_conv);
    last_elem = instance_bin->msg_conv;
  }



  if (config->tiled_display_config.enable) {
    if (config->tiled_display_config.columns *
        config->tiled_display_config.rows < config->num_source_sub_bins) {
      if (config->tiled_display_config.columns == 0) {
        config->tiled_display_config.columns =
            (guint) (sqrt (config->num_source_sub_bins) + 0.5);
      }
      config->tiled_display_config.rows =
          (guint) ceil (1.0 * config->num_source_sub_bins /
          config->tiled_display_config.columns);
      NVGSTDS_WARN_MSG_V
          ("Num of Tiles less than number of sources, readjusting to "
          "%u rows, %u columns", config->tiled_display_config.rows,
          config->tiled_display_config.columns);
    }

    if (!create_tiled_display_bin (&config->tiled_display_config,
            &pipeline->tiled_display_bin)) {
      g_print ("creating tiled display bin failed\n");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), pipeline->tiled_display_bin.bin);
    
    if(config->show_source != -1){
      //default -1 means composite and show all sources
      g_object_set(G_OBJECT(pipeline->tiled_display_bin.tiler), "show-source", config->show_source, NULL);
      g_print("show-source:%d\n", config->show_source);
    }


    NVGSTDS_LINK_ELEMENT (last_elem, pipeline->tiled_display_bin.bin);
    last_elem = pipeline->tiled_display_bin.bin;
    osd_sink_pad = gst_element_get_static_pad(pipeline->tiled_display_bin.tiler, "sink");
    NvDsAppPerfStructInt *str =  (NvDsAppPerfStructInt *)g_malloc0(sizeof(NvDsAppPerfStructInt));
    DemoPerfCtx *perf_ctx = (DemoPerfCtx *)g_malloc0(sizeof(DemoPerfCtx));
    g_mutex_init(&perf_ctx->fps_lock);
    str->context = perf_ctx;
    enable_perf_measurement (str, osd_sink_pad, config->num_source_sub_bins, 1, 0, perf_cb);
    gst_object_unref(osd_sink_pad);
  }

  if (config->osd_config.enable) {
    if (!create_osd_bin (&config->osd_config, &instance_bin->osd_bin)) {
      g_print ("creating osd bin failed\n");
      goto done;
    }
    gst_bin_add (GST_BIN (pipeline->pipeline), instance_bin->osd_bin.bin);
    NVGSTDS_LINK_ELEMENT (last_elem, instance_bin->osd_bin.bin);
    last_elem = instance_bin->osd_bin.bin;
    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad(instance_bin->osd_bin.nvosd, "sink");
    if (!osd_sink_pad)
      g_print("Unable to get sink pad\n");
    else {
      gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                        osd_sink_pad_buffer_probe, NULL, NULL);
      LatencyCtx *ctx = (LatencyCtx *)g_malloc0(sizeof(LatencyCtx));
      ctx->lock = (GMutex *)g_malloc0(sizeof(GMutex));
      ctx->num_sources = config->num_source_sub_bins;
      gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          latency_measurement_buf_prob, ctx, NULL);
    }
  }

  //create sink_tee
  instance_bin->sink_tee = gst_element_factory_make (NVDS_ELEM_TEE, "sink_tee");
  if (!instance_bin->sink_tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'sink_tee'");
    goto done;
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), instance_bin->sink_tee);
  NVGSTDS_LINK_ELEMENT (last_elem, instance_bin->sink_tee);
  last_elem = instance_bin->sink_tee;

  if (!create_sink_bin (config->num_sink_sub_bins,
        config->sink_bin_sub_bin_config, &instance_bin->sink_bin, 0)) {
    g_print ("creating sink bin failed\n");
    goto done;
  }
  //x264enc will output one buffer after input 66 buffers at default, enable zerolatency property.
  for(int i = 0; i < config->num_sink_sub_bins; i++){
      if(config->sink_bin_sub_bin_config[i].encoder_config.enc_type == NV_DS_ENCODER_TYPE_SW)
        g_object_set (G_OBJECT (instance_bin->sink_bin.sub_bins[i].encoder), "tune", 0x4, NULL);
  }
  gst_bin_add (GST_BIN (pipeline->pipeline), instance_bin->sink_bin.bin);
  NVGSTDS_LINK_ELEMENT (last_elem, instance_bin->sink_bin.bin);
 
  //link broker to sink-tee
  add_and_link_broker_sink(appCtx);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline->pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, main_loop);
  gst_object_unref(bus);

  /* Set the pipeline to "playing" state */
  gst_element_set_state(pipeline->pipeline, GST_STATE_PLAYING);

  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline->pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(main_loop);

done:

  g_print ("Quitting\n");
  if (bus_watch_id) {
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline->pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline->pipeline));
    g_source_remove(bus_watch_id);
  }

  if (appCtx) {
    if (appCtx->return_value == -1)
      return_value = -1;
    g_free (appCtx);
  }

  if (main_loop) {
    g_main_loop_unref (main_loop);
  }

  if (ctx) {
    g_option_context_free (ctx);
  }

  if (return_value == 0) {
    g_print ("App run successful\n");
  } else {
    g_print ("App run failed\n");
  }

  gst_deinit ();

  return return_value;
}

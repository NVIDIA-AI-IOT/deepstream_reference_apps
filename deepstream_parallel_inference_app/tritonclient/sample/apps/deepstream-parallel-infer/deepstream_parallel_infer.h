/*
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

#ifndef __NVGSTDS_APP_H__
#define __NVGSTDS_APP_H__

#include <gst/gst.h>
#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

#include "deepstream_app_version.h"
#include "deepstream_common.h"
#include "deepstream_config.h"
#include "deepstream_osd.h"
#include "deepstream_perf.h"
#include "deepstream_preprocess.h"
#include "deepstream_primary_gie.h"
#include "deepstream_sinks.h"
#include "deepstream_sources.h"
#include "deepstream_streammux.h"
#include "deepstream_tiled_display.h"
#include "deepstream_dsanalytics.h"
#include "deepstream_dsexample.h"
#include "deepstream_tracker.h"
#include "deepstream_secondary_gie.h"
#include "deepstream_c2d_msg.h"
#include "deepstream_image_save.h"

#define MAX_PRIMARY_GIE_BINS (16)
#define MAX_PRE_PROCESS_BINS (16)
#define MAX_VIDEO_TEMPLATE_PROPS (32)

typedef struct _AppCtx AppCtx;

typedef void (*bbox_generated_callback) (AppCtx *appCtx, GstBuffer *buf,
    NvDsBatchMeta *batch_meta, guint index);
typedef gboolean (*overlay_graphics_callback) (AppCtx *appCtx, GstBuffer *buf,
    NvDsBatchMeta *batch_meta, guint index);

typedef struct
{
  gboolean enable;

  gchar *config_file_path;
} NvDsMetaMuxConfig;

typedef struct
{
  gboolean enable;

  guint num_customlib_props;
  gchar *customlib_name;
  gchar *customlib_props[MAX_VIDEO_TEMPLATE_PROPS];
} NvDsVideoTemplateConfig;

typedef struct
{
  GstElement *bin;
  GstElement *tee;
  GstElement *muxer;
  gulong muxer_buffer_probe_id;
  GstElement *source_tee[MAX_PRIMARY_GIE_BINS];
  GstElement *demuxer;
  GstElement *streammux[MAX_PRIMARY_GIE_BINS];
  NvDsPrimaryGieBin primary_gie_bin[MAX_PRIMARY_GIE_BINS];
  NvDsTrackerBin tracker_bin[MAX_PRIMARY_GIE_BINS];
  NvDsSecondaryGieBin secondary_gie_bin[MAX_PRIMARY_GIE_BINS];
  NvDsDsAnalyticsBin dsanalytics_bin[MAX_PRIMARY_GIE_BINS];
  NvDsPreProcessBin preprocess_bin[MAX_PRIMARY_GIE_BINS];
} NvDsParallelGieBin;

typedef struct
{
  guint index;
  gulong all_bbox_buffer_probe_id;
  gulong primary_bbox_buffer_probe_id;
  gulong fps_buffer_probe_id;
  GstElement *bin;
  GstElement *tee;
  GstElement *sink_tee;
  GstElement *msg_conv;
  NvDsPreProcessBin preprocess_bin;
  NvDsPrimaryGieBin primary_gie_bin;
  NvDsOSDBin osd_bin;
  NvDsSecondaryGieBin secondary_gie_bin;
  NvDsTrackerBin tracker_bin;
  NvDsSinkBin sink_bin;
  NvDsSinkBin demux_sink_bin;
  NvDsDsAnalyticsBin dsanalytics_bin;
  NvDsDsExampleBin dsexample_bin;
  AppCtx *appCtx;
} NvDsInstanceBin;

typedef struct
{
  gulong primary_bbox_buffer_probe_id;
  guint bus_id;
  GstElement *pipeline;
  NvDsSrcParentBin multi_src_bin;
  NvDsParallelGieBin parallel_infer_bin;
  NvDsInstanceBin instance_bins[MAX_SOURCE_BINS];
  NvDsInstanceBin demux_instance_bins[MAX_SOURCE_BINS];
  NvDsInstanceBin common_elements;
  GstElement *tiler_tee;
  NvDsTiledDisplayBin tiled_display_bin;
  GstElement *demuxer;
  NvDsDsExampleBin dsexample_bin;
  AppCtx *appCtx;
} NvDsPipeline;

typedef struct
{
  gint pgie_id;
  gchar *src_ids;
} NvDsStrcIDConfig;

typedef struct
{
  gboolean enable_perf_measurement;
  gint file_loop;
  gint pipeline_recreate_sec;
  gboolean source_list_enabled;
  guint total_num_sources;
  guint num_source_sub_bins;
  guint num_secondary_gie_num[MAX_PRIMARY_GIE_BINS];
  guint num_secondary_gie_sub_bins;
  guint num_pre_process_sub_bins;
  guint num_primary_gie_sub_bins;
  guint num_src_ids_sub_bins;
  guint num_tracker_sub_bins;
  guint num_analysis_sub_bins;
  guint num_sink_sub_bins;
  guint num_message_consumers;
  guint perf_measurement_interval_sec;
  guint sgie_batch_size;
  gchar *bbox_dir_path;
  gchar *kitti_track_dir_path;
  guint show_source;

  gchar **uri_list;
  NvDsSourceConfig multi_source_config[MAX_SOURCE_BINS];
  NvDsStreammuxConfig streammux_config;
  NvDsOSDConfig osd_config;
  NvDsPreProcessConfig pre_process_sub_bin_config[MAX_PRIMARY_GIE_BINS];
  NvDsGieConfig primary_gie_sub_bin_config[MAX_PRIMARY_GIE_BINS];
  NvDsVideoTemplateConfig video_template_sub_bin_config[MAX_PRIMARY_GIE_BINS];
  NvDsMetaMuxConfig meta_mux_config;
  NvDsTrackerConfig tracker_config[MAX_PRIMARY_GIE_BINS];
  NvDsStrcIDConfig srcids_config[MAX_PRIMARY_GIE_BINS];
  NvDsGieConfig secondary_gie_sub_bin_config[MAX_PRIMARY_GIE_BINS][MAX_SECONDARY_GIE_BINS];
  NvDsSinkSubBinConfig sink_bin_sub_bin_config[MAX_SINK_BINS];
  NvDsMsgConsumerConfig message_consumer_config[MAX_MESSAGE_CONSUMERS];
  NvDsTiledDisplayConfig tiled_display_config;
  NvDsDsAnalyticsConfig dsanalytics_config[MAX_PRIMARY_GIE_BINS];
  NvDsDsExampleConfig dsexample_config;
  NvDsSinkMsgConvBrokerConfig msg_conv_config;
  NvDsImageSave image_save_config;

} NvDsConfig;

typedef struct
{
  gulong frame_num;
} NvDsInstanceData;

struct _AppCtx
{
  gboolean version;
  gboolean cintr;
  gboolean show_bbox_text;
  gboolean seeking;
  gboolean quit;
  gint person_class_id;
  gint car_class_id;
  gint return_value;
  guint index;
  gint active_source_index;

  GMutex app_lock;
  GCond app_cond;

  NvDsPipeline pipeline;
  NvDsConfig config;
  NvDsConfig override_config;
  NvDsInstanceData instance_data[MAX_SOURCE_BINS];
  NvDsC2DContext *c2d_ctx[MAX_MESSAGE_CONSUMERS];
  NvDsAppPerfStructInt perf_struct;
  bbox_generated_callback bbox_generated_post_analytics_cb;
  bbox_generated_callback all_bbox_generated_cb;
  overlay_graphics_callback overlay_graphics_cb;
  NvDsFrameLatencyInfo *latency_info;
  GMutex latency_lock;
  GThread *ota_handler_thread;
  guint ota_inotify_fd;
  guint ota_watch_desc;
};

/**
 * Function to read properties from YML configuration file.
 *
 * @param[in] config pointer to @ref NvDsConfig
 * @param[in] cfg_file_path path of configuration file.
 *
 * @return true if parsed successfully.
 */
gboolean
parse_config_file_yaml (NvDsConfig * config, gchar * cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif

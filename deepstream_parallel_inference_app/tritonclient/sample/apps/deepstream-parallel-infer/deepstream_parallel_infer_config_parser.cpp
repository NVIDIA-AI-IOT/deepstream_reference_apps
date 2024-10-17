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

#include <string>
#include <cstring>
#include "deepstream_parallel_infer.h"
#include "deepstream_config_yaml.h"
#include <iostream>

#include <stdlib.h>
#include <fstream>

using std::cout;
using std::endl;

static gboolean
parse_tests_yaml (NvDsConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["tests"].begin();
     itr != configyml["tests"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "file-loop") {
      config->file_loop = itr->second.as<gint>();
    } else {
      cout << "Unknown key " << paramKey << " for group tests" << endl;
    }
  }

  ret = TRUE;

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

static gboolean
parse_app_yaml (NvDsConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["application"].begin();
     itr != configyml["application"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable-perf-measurement") {
      config->enable_perf_measurement =
          itr->second.as<gboolean>();
    } else if (paramKey == "perf-measurement-interval-sec") {
      config->perf_measurement_interval_sec =
          itr->second.as<guint>();
    } else if (paramKey == "gie-kitti-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      config->bbox_dir_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->bbox_dir_path);
      g_free(str);
    } else if (paramKey == "kitti-track-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      config->kitti_track_dir_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->kitti_track_dir_path);
      g_free(str);
    }
    else {
      cout << "Unknown key " << paramKey << " for group application" << endl;
    }
  }

  ret = TRUE;

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

gboolean
parse_multi_preprocess_yaml (NvDsPreProcessConfig *config, std::string group, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml[group].begin();
     itr != configyml[group].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "config-file") {
      std::string temp = itr->second.as<std::string>();
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->config_file_path, temp.c_str(), 1024);
    } else {
      cout << "[WARNING] Unknown param found in pre-process: " << paramKey << endl;
    }
  }

  ret = TRUE;
  done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

gboolean
parse_video_template_yaml (NvDsVideoTemplateConfig *config, std::string group, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml[group].begin();
     itr != configyml[group].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "customlib-name") {
      std::string temp = itr->second.as<std::string>();
      config->customlib_name = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->customlib_name, temp.c_str(), 1024);
    } else if (paramKey == "customlib-props") {
      std::string temp = itr->second.as<std::string>();
      config->customlib_props[config->num_customlib_props] = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->customlib_props[config->num_customlib_props], temp.c_str(), 1024);
      config->num_customlib_props ++;
      if (config->num_customlib_props == MAX_VIDEO_TEMPLATE_PROPS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs", MAX_VIDEO_TEMPLATE_PROPS);
        ret = FALSE;
        goto done;
      }
    } else {
      cout << "[WARNING] Unknown param found in pre-process: " << paramKey << endl;
    }
  }

  ret = TRUE;
  done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

gboolean
parse_metamux_yaml (NvDsMetaMuxConfig *config, gchar* cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml["meta-mux"].begin();
     itr != configyml["meta-mux"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "config-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->config_file_path)) {
        g_printerr ("Error: Could not parse config-file-path in metamux.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else {
      cout << "[WARNING] Unknown param found in metamux: " << paramKey << endl;
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

static gboolean
parse_sgie_yaml (NvDsConfig *config, NvDsGieConfig *gieConfig, std::string group, gchar *cfg_file_path,
		gboolean enable)
{
  gboolean ret = FALSE;
  gboolean parse_err = false;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  int sgie_num = 0;
  gboolean have_one_enable = false;
  
  for(YAML::const_iterator itr = configyml.begin();
     itr != configyml.end(); ++itr)
  {
      std::string paramKey = itr->first.as<std::string>();
      parse_err =
          !parse_gie_yaml (gieConfig + sgie_num, paramKey, cfg_file_path);
      if(enable && gieConfig[sgie_num].enable){
         have_one_enable = true;
         sgie_num++;
      }
  }
  config->num_secondary_gie_num[config->num_secondary_gie_sub_bins] = sgie_num;
  config->num_secondary_gie_sub_bins++;
  
  ret = TRUE;
 

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

static std::vector<std::string>
split_csv_entries (std::string input) {
  std::vector<int> positions;
  for (unsigned int i = 0; i < input.size(); i++) {
    if (input[i] == ',')
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

gboolean
parse_tiled_display_yaml (NvDsConfig *appConfig, NvDsTiledDisplayConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml["tiled-display"].begin();
     itr != configyml["tiled-display"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable =
          (NvDsTiledDisplayEnable) itr->second.as<int>();
    } else if (paramKey == "rows") {
      config->rows = itr->second.as<guint>();
    } else if (paramKey == "columns") {
      config->columns = itr->second.as<guint>();
    } else if (paramKey == "width") {
      config->width = itr->second.as<guint>();
    } else if (paramKey == "height") {
      config->height = itr->second.as<guint>();
    } else if (paramKey == "gpu-id") {
      config->gpu_id = itr->second.as<guint>();
    } else if (paramKey == "nvbuf-memory-type") {
      config->nvbuf_memory_type = itr->second.as<guint>();
    } else if (paramKey == "show-source") {
      appConfig->show_source = itr->second.as<guint>();
    } else {
      cout << "[WARNING] Unknown param found in tiled-display: " << paramKey << endl;
    }
  }
  ret = TRUE;

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

gboolean
parse_config_file_yaml (NvDsConfig *config, gchar *cfg_file_path)
{
  gboolean parse_err = false;
  gboolean ret = FALSE;
  gboolean enable = false;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  std::string source_str = "source";
  std::string sink_str = "sink";
  std::string pre_process_str = "pre-process";
  std::string pgie_str = "primary-gie";
  std::string branch_str = "branch";
  std::string tracker_str = "tracker";
  std::string sgie_str = "secondary-gie";
  std::string video_template_str = "video-template";
  std::string msgcons_str = "message-consumer";
  std::string analytics_str = "nvds-analytics";
  config->source_list_enabled = FALSE;
  config->show_source = -1;  //default: show all source

  for(YAML::const_iterator itr = configyml.begin();
    itr != configyml.end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();

    if (paramKey ==  "application") {
      parse_err = !parse_app_yaml (config, cfg_file_path);
    }
    else if (paramKey == "source") {
      if(configyml["source"]["csv-file-path"]) {
        std::string csv_file_path = configyml["source"]["csv-file-path"].as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strncpy (str, csv_file_path.c_str(), 1024);
        char *abs_csv_path = (char*) malloc(sizeof(char) * 1024);
        get_absolute_file_path_yaml (cfg_file_path, str, abs_csv_path);
        g_free(str);

        std::ifstream inputFile (abs_csv_path);
        if (!inputFile.is_open()) {
          cout << "Couldn't open CSV file " << abs_csv_path << endl;
        }
        std::string line, temp;
        /* Separating header field and inserting as strings into the vector.
        */
        getline(inputFile, line);
        std::vector<std::string> headers = split_csv_entries(line);
        /*Parsing each csv entry as an input source */
        while(getline(inputFile, line)) {
          std::vector<std::string> source_values = split_csv_entries(line);
          if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
            NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
            ret = FALSE;
            goto done;
          }
          guint source_id = 0;
          source_id = config->num_source_sub_bins;
          parse_err = !parse_source_yaml (&config->multi_source_config[source_id], headers, source_values, cfg_file_path);
          if (config->multi_source_config[source_id].enable)
            config->num_source_sub_bins++;
        }
      } else {
        NVGSTDS_ERR_MSG_V ("CSV file not specified\n");
        ret = FALSE;
        goto done;
      }
    }
    else if (paramKey == "streammux") {
      parse_err = !parse_streammux_yaml(&config->streammux_config, cfg_file_path);
    }
    else if (paramKey == "osd") {
      parse_err = !parse_osd_yaml(&config->osd_config, cfg_file_path);
    }
    else if (paramKey.compare(0, pre_process_str.size(), pre_process_str) == 0) {
      if (config->num_pre_process_sub_bins == MAX_PRE_PROCESS_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d pre-process", MAX_PRE_PROCESS_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_multi_preprocess_yaml (&config->pre_process_sub_bin_config[config->
                                  num_pre_process_sub_bins],
                                  paramKey, cfg_file_path);
      if (config->pre_process_sub_bin_config[config->num_pre_process_sub_bins].enable){
        config->num_pre_process_sub_bins++;
      }
    }
    else if (paramKey.compare(0, pgie_str.size(), pgie_str) == 0) {
      if (config->num_primary_gie_sub_bins == MAX_PRIMARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs", MAX_PRIMARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_gie_yaml (&config->primary_gie_sub_bin_config[config->
                                  num_primary_gie_sub_bins],
                                  paramKey, cfg_file_path);
      if (config->primary_gie_sub_bin_config[config->num_primary_gie_sub_bins].enable){
        config->num_primary_gie_sub_bins++;
      }
    }
    else if (paramKey.compare(0, video_template_str.size(), video_template_str) == 0) {
      if (config->num_primary_gie_sub_bins == MAX_PRIMARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs", MAX_PRIMARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_video_template_yaml (&config->video_template_sub_bin_config[config->
                                  num_primary_gie_sub_bins],
                                  paramKey, cfg_file_path);
      if (config->video_template_sub_bin_config[config->num_primary_gie_sub_bins].enable){
        config->num_primary_gie_sub_bins++;
      }
    }
    else if (paramKey == "meta-mux") {
      parse_err = !parse_metamux_yaml (&config->meta_mux_config, cfg_file_path);
    }
    else if (paramKey.compare(0, branch_str.size(), branch_str) == 0) {
      if(configyml[paramKey]["pgie-id"].as<int>()){
          config->srcids_config[config->num_src_ids_sub_bins].pgie_id =
		  configyml[paramKey]["pgie-id"].as<int>();
          if(configyml[paramKey]["src-ids"]) {
              std::string src_ids = configyml[paramKey]["src-ids"].as<std::string>();
              int index = config->num_src_ids_sub_bins;
	      config->srcids_config[index].src_ids = (char*) calloc(sizeof(char) * src_ids.size(), sizeof(char));
	      std::strncpy (config->srcids_config[index].src_ids , src_ids.c_str(), src_ids.size());
	      g_print("src_ids:%s\n",  config->srcids_config[index].src_ids);
         }
      }
      config->num_src_ids_sub_bins++;
    }
    else if (paramKey.compare(0, tracker_str.size(), tracker_str) == 0) {
      if (config->num_tracker_sub_bins == MAX_PRIMARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d tracker", MAX_PRIMARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }

      enable = configyml[paramKey]["enable"].as<gboolean>();
      if(configyml[paramKey]["cfg-file-path"]) {
        std::string csv_file_path = configyml[paramKey]["cfg-file-path"].as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strncpy (str, csv_file_path.c_str(), 1024);

        char *abs_csv_path = (char*) malloc(sizeof(char) * 1024);
        get_absolute_file_path_yaml (cfg_file_path, str, abs_csv_path);
        g_free(str);

        parse_err =
            !parse_tracker_yaml (&config->tracker_config[config->
                                  num_tracker_sub_bins], abs_csv_path);
        g_free(abs_csv_path);
        config->tracker_config[config->num_tracker_sub_bins].enable = enable;
        config->num_tracker_sub_bins++;
      }
    }
    else if (paramKey.compare(0, sgie_str.size(), sgie_str) == 0) {
      if (config->num_secondary_gie_sub_bins == MAX_SECONDARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs", MAX_SECONDARY_GIE_BINS);
        ret = FALSE;
        goto done;
      } 
      enable = configyml[paramKey]["enable"].as<gboolean>();
      if( configyml[paramKey]["cfg-file-path"]) {
        std::string csv_file_path = configyml[paramKey]["cfg-file-path"].as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strncpy (str, csv_file_path.c_str(), 1024);
        char *abs_csv_path = (char*) malloc(sizeof(char) * 1024);
        get_absolute_file_path_yaml (cfg_file_path, str, abs_csv_path);
        g_free(str);
        NvDsGieConfig* pCfg = config->secondary_gie_sub_bin_config[config->
                              num_secondary_gie_sub_bins];
        parse_err =
          !parse_sgie_yaml(config, pCfg,
                              paramKey, abs_csv_path, enable);
      }
    }
    else if (paramKey.compare(0, sink_str.size(), sink_str) == 0) {
      if (config->num_sink_sub_bins == MAX_SINK_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sinks", MAX_SINK_BINS);
        ret = FALSE;
        goto done;
      }
      parse_err =
          !parse_sink_yaml (&config->
          sink_bin_sub_bin_config[config->num_sink_sub_bins], paramKey, cfg_file_path);
      if (config->
          sink_bin_sub_bin_config[config->num_sink_sub_bins].enable) {
        config->num_sink_sub_bins++;
      }
    }
    else if (paramKey.compare(0, msgcons_str.size(), msgcons_str) == 0) {
      if (config->num_message_consumers == MAX_MESSAGE_CONSUMERS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d consumers", MAX_MESSAGE_CONSUMERS);
        ret = FALSE;
        goto done;
      }
      parse_err = !parse_msgconsumer_yaml (
                    &config->message_consumer_config[config->num_message_consumers],
                    paramKey, cfg_file_path);

      if (config->message_consumer_config[config->num_message_consumers].enable) {
        config->num_message_consumers++;
      }
    }
    else if (paramKey == "tiled-display") {
      parse_err = !parse_tiled_display_yaml (config, &config->tiled_display_config, cfg_file_path);
    }
    else if (paramKey == "img-save") {
      parse_err = !parse_image_save_yaml (&config->image_save_config , cfg_file_path);
    }

    else if (paramKey == "ds-example") {
      parse_err = !parse_dsexample_yaml (&config->dsexample_config, cfg_file_path);
    }
    else if (paramKey == "message-converter") {
      parse_err = !parse_msgconv_yaml (&config->msg_conv_config, paramKey, cfg_file_path);
    }
    else if (paramKey.compare(0, analytics_str.size(), analytics_str) == 0) {
      if (config->num_analysis_sub_bins == MAX_PRIMARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d analysis", MAX_PRIMARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }
      enable = configyml[paramKey]["enable"].as<gboolean>();
      config->dsanalytics_config[config->num_analysis_sub_bins].enable = enable;
      if(configyml[paramKey]["cfg-file-path"]) {
        std::string temp = configyml[paramKey]["cfg-file-path"].as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strncpy (str, temp.c_str(), 1024);
        config->dsanalytics_config[config->num_analysis_sub_bins].config_file_path = 
        (char*) malloc(sizeof(char) * 1024);
        if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->dsanalytics_config[config->num_analysis_sub_bins].config_file_path)) {
          g_printerr ("Error: Could not parse config-file in dsanalytics.\n");
          g_free (str);
          goto done;
        }
        g_free (str);
      } else {
        cout << "[WARNING] Unknown param found in nvds-analytics: " << paramKey << endl;
      }
      config->num_analysis_sub_bins++;
    }
    else if (paramKey == "tests") {
      parse_err = !parse_tests_yaml (config, cfg_file_path);
    }

    if (parse_err) {
      cout << "failed parsing" << endl;
      goto done;
    }
  }
  /* Updating batch size when source list is enabled */
  /* if (config->source_list_enabled == TRUE) {
      // For streammux and pgie, batch size is set to number of sources
      config->streammux_config.batch_size = config->num_source_sub_bins;
      config->primary_gie_config.batch_size = config->num_source_sub_bins;
      if (config->sgie_batch_size != 0) {
          for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
              config->secondary_gie_sub_bin_config[i].batch_size = config->sgie_batch_size;
          }
      }
  } */
 
   unsigned int i, j, k;
   for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    for(int j = 0; j < config->num_secondary_gie_num[i]; j++ ){
      if (config->secondary_gie_sub_bin_config[i][j].unique_id ==
          config->primary_gie_sub_bin_config[i].unique_id) {
        NVGSTDS_ERR_MSG_V ("Non unique gie ids found");
        ret = FALSE;
        goto done;
      }
    }
  }

  for (k = 0; k < config->num_secondary_gie_sub_bins; k++) {
    for (i = 0; i < config->num_secondary_gie_num[k]; i++) {
      for (j = i + 1; j < config->num_secondary_gie_num[k]; j++) {
        if (config->secondary_gie_sub_bin_config[k][i].unique_id ==
            config->secondary_gie_sub_bin_config[k][j].unique_id) {
          NVGSTDS_ERR_MSG_V ("Non unique gie id %d found",
                              config->secondary_gie_sub_bin_config[k][i].unique_id);
          ret = FALSE;
          goto done;
        }
      }
    }
  }
  
  for (i = 0; i < config->num_source_sub_bins; i++) {
    if (config->multi_source_config[i].type == NV_DS_SOURCE_URI_MULTIPLE) {
      if (config->multi_source_config[i].num_sources < 1) {
        config->multi_source_config[i].num_sources = 1;
      }
      for (j = 1; j < config->multi_source_config[i].num_sources; j++) {
        if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
          NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
          ret = FALSE;
          goto done;
        }
        memcpy (&config->multi_source_config[config->num_source_sub_bins],
            &config->multi_source_config[i],
            sizeof (config->multi_source_config[i]));
        config->multi_source_config[config->num_source_sub_bins].type =
            NV_DS_SOURCE_URI;
        config->multi_source_config[config->num_source_sub_bins].uri =
            g_strdup_printf (config->multi_source_config[config->
                num_source_sub_bins].uri, j);
        config->num_source_sub_bins++;
      }
      config->multi_source_config[i].type = NV_DS_SOURCE_URI;
      config->multi_source_config[i].uri =
          g_strdup_printf (config->multi_source_config[i].uri, 0);
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

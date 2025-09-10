/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "yaml_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <assert.h>
#include "cuda_runtime_api.h"
 #include <cstring>

using std::endl;
using std::cout;
 
static gboolean
gst_parse_props_yaml (const gchar * cfg_file_path, cfg_params& cfg_params)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }
  for(YAML::const_iterator itr = configyml["property"].begin(); itr != configyml["property"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "width") {
        cfg_params.m_tensor_width = itr->second.as<unsigned int>();
    } else if (paramKey == "height") {
        cfg_params.m_tensor_height = itr->second.as<unsigned int>();
    } else {
      std::string paramVal = itr->second.as<std::string>();
      printf("not need %s\n", paramVal.c_str());
    }
  }

  ret = TRUE;
done:
  return ret;
}

/* Parse nvinfer config file for context params. Returns FALSE in case of an error. */
gboolean
gst_parse_context_params_yaml (const gchar * cfg_file_path, cfg_params& cfg_params)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }
  /* 'property' group is mandatory. */
  if(configyml["property"]) {
    if (!gst_parse_props_yaml (cfg_file_path, cfg_params)) {
      g_printerr ("Failed to parse group property\n");
      goto done;
    }
  }
  else  {
    g_printerr ("Could not find group property\n");
    goto done;
  }
  ret = TRUE;

done:
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}

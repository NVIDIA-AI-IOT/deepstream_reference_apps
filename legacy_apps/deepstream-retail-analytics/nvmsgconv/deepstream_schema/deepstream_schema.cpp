/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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



#include "deepstream_schema.h"
#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

using namespace std;

static void
get_csv_tokens (const string &text, vector<string> &tokens)
{
  /* This is based on assumption that fields and their locations
   * are fixed in CSV file. This should be updated accordingly if
   * that is not the case.
   */
  gint count = 0;

  gchar **csv_tokens = g_strsplit (text.c_str(), ",", -1);
  gchar **temp = csv_tokens;
  gchar *token;

  while (*temp && count < DEFAULT_CSV_FIELDS) {
    token = *temp++;
    tokens.push_back (string(g_strstrip(token)));
    count++;
  }
  g_strfreev (csv_tokens);
}

static bool
nvds_msg2p_parse_sensor (void *privData, GKeyFile *key_file, gchar *group)
{
  bool ret = false;
  bool isEnabled = false;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject sensorObj;
  gint sensorId;
  gchar *keyVal;


  if (sscanf (group, CONFIG_GROUP_SENSOR "%u", &sensorId) < 1) {
    cout << "Wrong sensor group name " << group << endl;
    return ret;
  }

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->sensorObj.find (sensorId);
  if (idMap != privObj->sensorObj.end()) {
    cout << "Duplicate entries for " << group << endl;
    return ret;
  }

  isEnabled = g_key_file_get_boolean (key_file, group, CONFIG_KEY_ENABLE,
                                      &error);
  if (!isEnabled) {
    // Not enabled, skip the parsing of keys.
    ret = true;
    goto done;
  } else {
    g_key_file_remove_key (key_file, group, CONFIG_KEY_ENABLE,
                           &error);
    CHECK_ERROR (error);
  }

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    keyVal = NULL;
    if (!g_strcmp0 (*key, CONFIG_KEY_ID)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_ID, &error);
      sensorObj.id = keyVal;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_TYPE)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_TYPE, &error);
      sensorObj.type = keyVal;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_DESCRIPTION)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_DESCRIPTION, &error);
      sensorObj.desc = keyVal;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_LOCATION)) {
      gsize length;
      gdouble *location = g_key_file_get_double_list (key_file, group,
                                                      CONFIG_KEY_LOCATION,
                                                      &length, &error);
      if (length != 3) {
        cout << "Wrong values provided, it should be like lat;lon;alt" << endl;
        g_free (location);
        goto done;
      }

      memcpy (sensorObj.location, location, length * sizeof (gdouble));
      g_free (location);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_COORDINATE)) {
      gsize length;
      gdouble *coordinate = g_key_file_get_double_list (key_file, group,
                                                      CONFIG_KEY_COORDINATE,
                                                      &length, &error);
      if (length != 3) {
        cout << "Wrong values provided, it should be like x;y;z" << endl;
        g_free (coordinate);
        goto done;
      }

      memcpy (sensorObj.coordinate, coordinate, length * sizeof (gdouble));
      g_free (coordinate);
      CHECK_ERROR (error);
    } else {
      cout << "Unknown key " << *key << " for group [" << group <<"]\n";
    }

    if (keyVal)
      g_free (keyVal);
  }

  privObj->sensorObj.insert (make_pair (sensorId, sensorObj));

  ret = true;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }

  return ret;
}

static bool
nvds_msg2p_parse_place (void *privData, GKeyFile *key_file, gchar *group)
{
  bool ret = false;
  bool isEnabled = false;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  NvDsPayloadPriv *privObj = NULL;
  NvDsPlaceObject placeObj;
  gint placeId;
  gchar *keyVal;

  if (sscanf (group, CONFIG_GROUP_PLACE "%u", &placeId) < 1) {
    cout << "Wrong place group name " << group << endl;
    return ret;
  }

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->placeObj.find (placeId);
  if (idMap != privObj->placeObj.end()) {
    cout << "Duplicate entries for " << group << endl;
    return ret;
  }

  isEnabled = g_key_file_get_boolean (key_file, group, CONFIG_KEY_ENABLE,
                                      &error);
  if (!isEnabled) {
    // Not enabled, skip the parsing of keys.
    ret = true;
    goto done;
  } else {
    g_key_file_remove_key (key_file, group, CONFIG_KEY_ENABLE,
                           &error);
    CHECK_ERROR (error);
  }

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_KEY_ID)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_ID, &error);
      placeObj.id = keyVal;
      g_free (keyVal);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_TYPE)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_TYPE, &error);
      placeObj.type = keyVal;
      g_free (keyVal);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_NAME)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_NAME, &error);
      placeObj.name = keyVal;
      g_free (keyVal);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_LOCATION)) {
      gsize length;
      gdouble *location = g_key_file_get_double_list (key_file, group,
                                                      CONFIG_KEY_LOCATION,
                                                      &length, &error);
      if (length != 3) {
        cout << "Wrong values provided, it should be like lat;lon;alt" << endl;
        g_free (location);
        goto done;
      }

      memcpy (placeObj.location, location, length * sizeof (gdouble));
      g_free (location);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_COORDINATE)) {
      gsize length;
      gdouble *coordinate = g_key_file_get_double_list (key_file, group,
                                                      CONFIG_KEY_COORDINATE,
                                                      &length, &error);
      if (length != 3) {
        cout << "Wrong values provided, it should be like x;y;z" << endl;
        g_free (coordinate);
        goto done;
      }

      memcpy (placeObj.coordinate, coordinate, length * sizeof (gdouble));
      g_free (coordinate);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_PLACE_SUB_FIELD1)) {
      keyVal = g_key_file_get_string (key_file, group,
                                        CONFIG_KEY_PLACE_SUB_FIELD1, &error);
      placeObj.subObj.field1 = keyVal;
      g_free (keyVal);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_PLACE_SUB_FIELD2)) {
      keyVal = g_key_file_get_string (key_file, group,
                                        CONFIG_KEY_PLACE_SUB_FIELD2, &error);
      placeObj.subObj.field2 = keyVal;
      g_free (keyVal);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_PLACE_SUB_FIELD3)) {
      keyVal = g_key_file_get_string (key_file, group,
                                        CONFIG_KEY_PLACE_SUB_FIELD3, &error);
      placeObj.subObj.field3 = keyVal;
      g_free (keyVal);
      CHECK_ERROR (error);
    } else {
      cout << "Unknown key " << *key << " for group [" << group <<"]\n";
    }
  }

  privObj->placeObj.insert (pair<int, NvDsPlaceObject> (placeId, placeObj));

  ret = true;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }

  return ret;
}

static bool
nvds_msg2p_parse_analytics (void *privData, GKeyFile *key_file, gchar *group)
{
  bool ret = false;
  bool isEnabled = false;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  NvDsPayloadPriv *privObj = NULL;
  NvDsAnalyticsObject analyticsObj;
  gint moduleId;
  gchar *keyVal;

  if (sscanf (group, CONFIG_GROUP_ANALYTICS "%u", &moduleId) < 1) {
    cout << "Wrong analytics module group name " << group << endl;
    return ret;
  }

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->analyticsObj.find (moduleId);
  if (idMap != privObj->analyticsObj.end()) {
    cout << "Duplicate entries for " << group << endl;
    return ret;
  }

  isEnabled = g_key_file_get_boolean (key_file, group, CONFIG_KEY_ENABLE,
                                      &error);
  if (!isEnabled) {
    // Not enabled, skip the parsing of keys.
    ret = true;
    goto done;
  } else {
    g_key_file_remove_key (key_file, group, CONFIG_KEY_ENABLE,
                           &error);
    CHECK_ERROR (error);
  }

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    keyVal = NULL;
    if (!g_strcmp0 (*key, CONFIG_KEY_ID)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_ID, &error);
      analyticsObj.id = keyVal;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_SOURCE)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_SOURCE, &error);
      analyticsObj.source = keyVal;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_DESCRIPTION)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_DESCRIPTION, &error);
      analyticsObj.desc = keyVal;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_KEY_VERSION)) {
      keyVal = g_key_file_get_string (key_file, group,
                                      CONFIG_KEY_VERSION, &error);
      analyticsObj.version = keyVal;
      CHECK_ERROR (error);
    } else {
      cout << "Unknown key " << *key << " for group [" << group <<"]\n";
    }

    if (keyVal)
      g_free (keyVal);
  }

  privObj->analyticsObj.insert (make_pair (moduleId, analyticsObj));

  ret = true;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }

  return ret;
}

/* Separate a config file entry with delimiters
 * into strings. */
static std::vector<std::string>
split_string (std::string input) {
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

static bool
nvds_msg2p_parse_sensor_yaml (void *privData, gchar *cfg_file_path, std::string group_str)
{
  bool ret = false;
  bool isEnabled = false;
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject sensorObj;
  gint sensorId;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  if (sscanf (group_str.c_str(), CONFIG_GROUP_SENSOR "%u", &sensorId) < 1) {
    cout << "Wrong sensor group format " << group_str << endl;
    return ret;
  }

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->sensorObj.find (sensorId);
  if (idMap != privObj->sensorObj.end()) {
    cout << "Duplicate entries for " << group_str << endl;
    return ret;
  }

  if (configyml[group_str]["enable"]) {
    isEnabled = configyml[group_str]["enable"].as<gboolean>();
    if(isEnabled == FALSE) {
      ret = true;
      goto done;
    }
  } else {
    ret = true;
    goto done;
  }

  for(YAML::const_iterator itr = configyml[group_str].begin();
      itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      continue;
    } else if (paramKey == "id") {
      sensorObj.id = itr->second.as<std::string>();
    } else if (paramKey == "type") {
      sensorObj.type = itr->second.as<std::string>();
    } else if (paramKey == "description") {
      sensorObj.desc = itr->second.as<std::string>();
    } else if (paramKey == "location") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 3) {
        cout << "Wrong values provided, it should be like lat;lon;alt" << endl;
        goto done;
      }
      for(int i = 0; i < 3; i++) {
        sensorObj.location[i] = std::stod(vec[i]);
      }
    } else if (paramKey == "coordinate") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 3) {
        cout << "Wrong values provided, it should be like x;y;z" << endl;
        goto done;
      }
      for(int i = 0; i < 3; i++) {
        sensorObj.coordinate[i] = std::stod(vec[i]);
      }
    } else {
      cout << "Unknown key " << paramKey << " for group [" << group_str << "]\n";
    }
  }
  privObj->sensorObj.insert (make_pair (sensorId, sensorObj));
  ret = true;

done:
  return ret;
}

static bool
nvds_msg2p_parse_place_yaml (void *privData, gchar *cfg_file_path, std::string group_str)
{
  bool ret = false;
  bool isEnabled = false;
  NvDsPayloadPriv *privObj = NULL;
  NvDsPlaceObject placeObj;
  gint placeId;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  if (sscanf (group_str.c_str(), CONFIG_GROUP_PLACE "%u", &placeId) < 1) {
    cout << "Wrong place group name " << group_str << endl;
    return ret;
  }

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->placeObj.find (placeId);
  if (idMap != privObj->placeObj.end()) {
    cout << "Duplicate entries for " << group_str << endl;
    return ret;
  }

  if (configyml[group_str]["enable"]) {
    isEnabled = configyml[group_str]["enable"].as<gboolean>();
    if(isEnabled == FALSE) {
      ret = true;
      goto done;
    }
  } else {
    ret = true;
    goto done;
  }

  for(YAML::const_iterator itr = configyml[group_str].begin();
      itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      continue;
    } else if (paramKey ==  "id") {
      placeObj.id = itr->second.as<std::string>();
    } else if (paramKey == "type") {
      placeObj.type = itr->second.as<std::string>();
    } else if (paramKey ==  "name") {
      placeObj.name = itr->second.as<std::string>();
    } else if (paramKey == "location") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 3) {
        cout << "Wrong values provided, it should be like lat;lon;alt" << endl;
        goto done;
      }
      for(int i = 0; i < 3; i++) {
        placeObj.location[i] = std::stod(vec[i]);
      }
    } else if (paramKey == "coordinate") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 3) {
        cout << "Wrong values provided, it should be like x;y;z" << endl;
        goto done;
      }
      for(int i = 0; i < 3; i++) {
        placeObj.coordinate[i] = std::stod(vec[i]);
      }
    } else if (paramKey ==  "place-sub-field1") {
      placeObj.subObj.field1 = itr->second.as<std::string>();
    } else if (paramKey ==  "place-sub-field2") {
      placeObj.subObj.field2 = itr->second.as<std::string>();
    } else if (paramKey ==  "place-sub-field3") {
      placeObj.subObj.field3 = itr->second.as<std::string>();
    } else {
      cout << "Unknown key " << paramKey << " for group [" << group_str <<"]\n";
    }
  }
  privObj->placeObj.insert (pair<int, NvDsPlaceObject> (placeId, placeObj));
  ret = true;

done:
  return ret;
}

static bool
nvds_msg2p_parse_analytics_yaml (void *privData, gchar *cfg_file_path, std::string group_str)
{
  bool ret = false;
  bool isEnabled = false;
  NvDsPayloadPriv *privObj = NULL;
  NvDsAnalyticsObject analyticsObj;
  gint moduleId;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  if (sscanf (group_str.c_str(), CONFIG_GROUP_ANALYTICS "%u", &moduleId) < 1) {
    cout << "Wrong analytics module group name " << group_str << endl;
    return ret;
  }

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->analyticsObj.find (moduleId);
  if (idMap != privObj->analyticsObj.end()) {
    cout << "Duplicate entries for " << group_str << endl;
    return ret;
  }

  if (configyml[group_str]["enable"]) {
    isEnabled = configyml[group_str]["enable"].as<gboolean>();
    if(isEnabled == FALSE) {
      ret = true;
      goto done;
    }
  } else {
    ret = true;
    goto done;
  }

 for(YAML::const_iterator itr = configyml[group_str].begin();
      itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      continue;
    } else if (paramKey == "id") {
      analyticsObj.id = itr->second.as<std::string>();
    } else if (paramKey == "source") {
      analyticsObj.source = itr->second.as<std::string>();
    } else if (paramKey == "description") {
      analyticsObj.desc = itr->second.as<std::string>();
    } else if (paramKey == "version") {
      analyticsObj.version = itr->second.as<std::string>();
    } else {
      cout << "Unknown key " << paramKey << " for group [" << group_str <<"]\n";
    }
  }
  privObj->analyticsObj.insert (make_pair (moduleId, analyticsObj));
  ret = true;

done:
  return ret;
}

bool nvds_msg2p_parse_csv (void *privData, const gchar *file)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsAnalyticsObject analyticsObj;
  NvDsSensorObject sensorObj;
  NvDsPlaceObject placeObj;
  bool retVal = true;
  bool firstRow = true;
  string line;
  gint i, index = 0;

  ifstream inputFile (file);
  if (!inputFile.is_open()) {
    cout << "Couldn't open CSV file " << file << endl;
    return false;
  }

  privObj = (NvDsPayloadPriv *) privData;

  try {

    while (getline (inputFile, line)) {

      if (firstRow) {
        // Discard first row as it will have header fields.
        firstRow = false;
        continue;
      }

      vector<string> tokens;
      get_csv_tokens (line, tokens);
      // Ignore first cameraId field.
      i = 1;

      // sensor object fields
      sensorObj.id = tokens.at(i++);
      sensorObj.type = "Camera";
      sensorObj.desc = tokens.at(i++);

      //Hard coded values but can be read from CSV file.
      sensorObj.location[0] = 0; //atof (tokens.at(i++).c_str ());
      sensorObj.location[1] = 0;
      sensorObj.location[2] = 0;
      sensorObj.coordinate[0] = 0;
      sensorObj.coordinate[1] = 0;
      sensorObj.coordinate[2] = 0;

      // place object fields
      placeObj.id = "Id";
      placeObj.type = "building/garage";
      placeObj.name = "endeavor";
      placeObj.location[0] = 0;
      placeObj.location[1] = 0;
      placeObj.location[2] = 0;
      placeObj.coordinate[0] = 0;
      placeObj.coordinate[1] = 0;
      placeObj.coordinate[2] = 0;
      //Ignore cameraIDstring
      i++;
      placeObj.subObj.field1 = tokens.at(i++);
      placeObj.subObj.field2 = tokens.at(i++);
      placeObj.subObj.field3 = tokens.at(i++);

      // analytics object fields
      // hard coded values but can be read from CSV file.
      analyticsObj.id = "";
      analyticsObj.source = "";
      analyticsObj.desc = "";
      analyticsObj.version = "1.0";

      privObj->sensorObj.insert (make_pair (index, sensorObj));
      privObj->placeObj.insert (make_pair (index, placeObj));
      privObj->analyticsObj.insert (make_pair (index, analyticsObj));

      index++;
    }
  } catch (const std::out_of_range& oor) {
    std::cerr << "Out of Range error: " << oor.what() << '\n';
    retVal = false;
  }

  inputFile.close ();
  return retVal;
}

bool nvds_msg2p_parse_yaml (void *privData, const gchar *file)
{
  bool retVal = true;
  YAML::Node configyml = YAML::LoadFile(file);
  std::string sensor_str = "sensor";
  std::string place_str = "place";
  std::string analytics_str = "analytics";
  gchar *cfg_file = (gchar *) malloc(sizeof(gchar *));
  cfg_file =  (gchar *) file;

  for(YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();

    if (paramKey.compare(0, sensor_str.size(), sensor_str) == 0) {
      retVal = nvds_msg2p_parse_sensor_yaml (privData, cfg_file, paramKey);
    } else if (paramKey.compare(0, place_str.size(), place_str) == 0) {
      retVal = nvds_msg2p_parse_place_yaml (privData, cfg_file, paramKey);
    } else if (paramKey.compare(0, analytics_str.size(), analytics_str) == 0) {
      retVal = nvds_msg2p_parse_analytics_yaml (privData, cfg_file, paramKey);
    } else {
      cout << "Unknown group " << paramKey << endl;
    }

    if (!retVal) {
      cout << "Failed to parse group " << paramKey << endl;
      goto done;
    }
  }

done:
  return retVal;
}

bool nvds_msg2p_parse_key_value (void *privData, const gchar *file)
{
  bool retVal = true;
  GKeyFile *cfgFile = NULL;
  GError *error = NULL;
  gchar **groups = NULL;
  gchar **group;

  cfgFile = g_key_file_new ();
  if (!g_key_file_load_from_file (cfgFile, file, G_KEY_FILE_NONE, &error)) {
    g_message ("Failed to load file: %s", error->message);
    retVal = false;
    goto done;
  }

  groups = g_key_file_get_groups (cfgFile, NULL);

  for (group = groups; *group; group++) {
    if (!strncmp (*group, CONFIG_GROUP_SENSOR, strlen (CONFIG_GROUP_SENSOR))) {
      retVal = nvds_msg2p_parse_sensor (privData, cfgFile, *group);
    } else if (!strncmp (*group, CONFIG_GROUP_PLACE, strlen (CONFIG_GROUP_PLACE))) {
      retVal = nvds_msg2p_parse_place (privData, cfgFile, *group);
    } else if (!strncmp (*group, CONFIG_GROUP_ANALYTICS, strlen (CONFIG_GROUP_ANALYTICS))) {
      retVal = nvds_msg2p_parse_analytics (privData, cfgFile, *group);
    } else {
      cout << "Unknown group " << *group << endl;
    }

    if (!retVal) {
      cout << "Failed to parse group " << *group << endl;
      goto done;
    }
  }

done:
  if (groups)
    g_strfreev (groups);

  if (cfgFile)
    g_key_file_free (cfgFile);

  return retVal;
}

void *create_deepstream_schema_ctx() {
    return (void *) new NvDsPayloadPriv;
}

void destroy_deepstream_schema_ctx(void *ptr) {
    NvDsPayloadPriv *privObj = (NvDsPayloadPriv *) ptr;
    delete privObj;
}

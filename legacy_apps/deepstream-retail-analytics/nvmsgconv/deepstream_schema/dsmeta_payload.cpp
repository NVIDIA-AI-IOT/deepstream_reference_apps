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


#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "deepstream_schema.h"

using namespace std;

#define MAX_TIME_STAMP_LEN (64)

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

static JsonObject* generate_place_object (void *privData, NvDsFrameMeta *frame_meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsPlaceObject *dsPlaceObj = NULL;
  JsonObject *placeObj;
  JsonObject *jobject;
  JsonObject *jobject2;

  privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->placeObj.find (frame_meta->source_id);

  if (idMap != privObj->placeObj.end()) {
    dsPlaceObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_PLACE << frame_meta->source_id
        << " in configuration file" << endl;
    return NULL;
  }

  /* place object
   * "place":
     {
       "id": "string",
       "name": "endeavor",
       “type”: “garage”,
       "location": {
         "lat": 30.333,
         "lon": -40.555,
         "alt": 100.00
       },
       "entrance/aisle": {
         "name": "walsh",
         "lane": "lane1",
         "level": "P2",
         "coordinate": {
           "x": 1.0,
           "y": 2.0,
           "z": 3.0
         }
       }
     }
   */

  placeObj = json_object_new ();
  json_object_set_string_member (placeObj, "id", dsPlaceObj->id.c_str());
  json_object_set_string_member (placeObj, "name", dsPlaceObj->name.c_str());
  json_object_set_string_member (placeObj, "type", dsPlaceObj->type.c_str());

  // location sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "lat", dsPlaceObj->location[0]);
  json_object_set_double_member (jobject, "lon", dsPlaceObj->location[1]);
  json_object_set_double_member (jobject, "alt", dsPlaceObj->location[2]);
  json_object_set_object_member (placeObj, "location", jobject);

  // place sub object (user to provide the name for sub place ex: parkingSpot/aisle/entrance..etc
  jobject = json_object_new ();

  json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
  json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field2.c_str());
  json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
  json_object_set_object_member (placeObj, "place-sub-field", jobject);

  // coordinates for place sub object
  jobject2 = json_object_new ();
  json_object_set_double_member (jobject2, "x", dsPlaceObj->coordinate[0]);
  json_object_set_double_member (jobject2, "y", dsPlaceObj->coordinate[1]);
  json_object_set_double_member (jobject2, "z", dsPlaceObj->coordinate[2]);
  json_object_set_object_member (jobject, "coordinate", jobject2);

  return placeObj;
}

static JsonObject* generate_sensor_object (void *privData, NvDsFrameMeta *frame_meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject *dsSensorObj = NULL;
  JsonObject *sensorObj;
  JsonObject *jobject;

  privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->sensorObj.find (frame_meta->source_id);

  if (idMap != privObj->sensorObj.end()) {
    dsSensorObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_SENSOR << frame_meta->source_id
         << " in configuration file" << endl;
    return NULL;
  }

  /* sensor object
   * "sensor": {
       "id": "string",
       "type": "Camera/Puck",
       "location": {
         "lat": 45.99,
         "lon": 35.54,
         "alt": 79.03
       },
       "coordinate": {
         "x": 5.2,
         "y": 10.1,
         "z": 11.2
       },
       "description": "Entrance of Endeavor Garage Right Lane"
     }
   */

  // sensor object
  sensorObj = json_object_new ();
  json_object_set_string_member (sensorObj, "id", dsSensorObj->id.c_str());
  json_object_set_string_member (sensorObj, "type", dsSensorObj->type.c_str());
  json_object_set_string_member (sensorObj, "description", dsSensorObj->desc.c_str());

  // location sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "lat", dsSensorObj->location[0]);
  json_object_set_double_member (jobject, "lon", dsSensorObj->location[1]);
  json_object_set_double_member (jobject, "alt", dsSensorObj->location[2]);
  json_object_set_object_member (sensorObj, "location", jobject);

  // coordinate sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "x", dsSensorObj->coordinate[0]);
  json_object_set_double_member (jobject, "y", dsSensorObj->coordinate[1]);
  json_object_set_double_member (jobject, "z", dsSensorObj->coordinate[2]);
  json_object_set_object_member (sensorObj, "coordinate", jobject);

  return sensorObj;
}

static JsonObject* generate_analytics_module_object (void *privData, NvDsFrameMeta *frame_meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsAnalyticsObject *dsObj = NULL;
  JsonObject *analyticsObj;

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->analyticsObj.find (frame_meta->source_id);

  if (idMap != privObj->analyticsObj.end()) {
    dsObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_ANALYTICS << frame_meta->source_id
        << " in configuration file" << endl;
    return NULL;
  }

  /* analytics object
   * "analyticsModule": {
       "id": "string",
       "description": "Vehicle Detection and License Plate Recognition",
       "confidence": 97.79,
       "source": "OpenALR",
       "version": "string"
     }
   */

  // analytics object
  analyticsObj = json_object_new ();
  json_object_set_string_member (analyticsObj, "id", dsObj->id.c_str());
  json_object_set_string_member (analyticsObj, "description", dsObj->desc.c_str());
  json_object_set_string_member (analyticsObj, "source", dsObj->source.c_str());
  json_object_set_string_member (analyticsObj, "version", dsObj->version.c_str());

  return analyticsObj;
}

static JsonObject*
generate_object_object (void *privData, NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
  JsonObject *objectObj;
  JsonObject *jobject;
  gchar tracking_id[64];
  //GList *objectMask = NULL;

  // object object
  objectObj = json_object_new ();
  if (snprintf (tracking_id, sizeof(tracking_id), "%lu", obj_meta->object_id)
      >= (int) sizeof(tracking_id))
    g_warning("Not enough space to copy trackingId");
  json_object_set_string_member (objectObj, "id", tracking_id);
  json_object_set_double_member (objectObj, "speed", 0);
  json_object_set_double_member (objectObj, "direction", 0);
  json_object_set_double_member (objectObj, "orientation", 0);

  jobject = json_object_new ();
  json_object_set_double_member (jobject, "confidence", obj_meta->confidence);

  //Fetch object classifiers detected
  for(NvDsClassifierMetaList *cl = obj_meta->classifier_meta_list; cl ; cl=cl->next) {
    NvDsClassifierMeta *cl_meta = (NvDsClassifierMeta*) cl->data;

    for(NvDsLabelInfoList *ll = cl_meta->label_info_list; ll ; ll=ll->next) {
        NvDsLabelInfo *ll_meta = (NvDsLabelInfo*) ll->data;
        if(cl_meta->classifier_type != NULL && strcmp("", cl_meta->classifier_type))
            json_object_set_string_member (jobject, cl_meta->classifier_type, ll_meta->result_label);
    }
  }
  json_object_set_object_member (objectObj, obj_meta->obj_label , jobject);

  // bbox sub object
  float scaleW = (float) frame_meta->source_frame_width /
                        (frame_meta->pipeline_width == 0) ? 1:frame_meta->pipeline_width;
  float scaleH = (float) frame_meta->source_frame_height /
                        (frame_meta->pipeline_height == 0) ? 1:frame_meta->pipeline_height;

  float left   = obj_meta->rect_params.left   * scaleW;
  float top    = obj_meta->rect_params.top    * scaleH;
  float width  = obj_meta->rect_params.width  * scaleW;
  float height = obj_meta->rect_params.height * scaleH;

  jobject = json_object_new ();
  json_object_set_int_member (jobject, "topleftx", left);
  json_object_set_int_member (jobject, "toplefty", top);
  json_object_set_int_member (jobject, "bottomrightx", left + width);
  json_object_set_int_member (jobject, "bottomrighty", top + height);
  json_object_set_object_member (objectObj, "bbox", jobject);

  // location sub object
  jobject = json_object_new ();
  json_object_set_object_member (objectObj, "location", jobject);

  // coordinate sub object
  jobject = json_object_new ();
  json_object_set_object_member (objectObj, "coordinate", jobject);

  return objectObj;
}

static JsonObject* generate_event_object (NvDsObjectMeta *obj_meta)
{
  JsonObject *eventObj;
  uuid_t uuid;
  gchar uuidStr[37];

  /*
   * "event": {
       "id": "event-id",
       "type": "entry / exit"
     }
   */

  uuid_generate_random (uuid);
  uuid_unparse_lower(uuid, uuidStr);

  eventObj = json_object_new ();
  json_object_set_string_member (eventObj, "id", uuidStr);
  json_object_set_string_member (eventObj, "type", "");
  return eventObj;
}

gchar* generate_dsmeta_message (void *privData, void *frameMeta, void *objMeta)
{
  JsonNode *rootNode;
  JsonObject *rootObj;
  JsonObject *placeObj;
  JsonObject *sensorObj;
  JsonObject *analyticsObj;
  JsonObject *eventObj;
  JsonObject *objectObj;
  gchar *message;

  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)frameMeta;
  NvDsObjectMeta *obj_meta  = (NvDsObjectMeta *)objMeta;

  uuid_t msgId;
  gchar msgIdStr[37];

  uuid_generate_random (msgId);
  uuid_unparse_lower(msgId, msgIdStr);

  // place object
  placeObj = generate_place_object (privData, frame_meta);

  // sensor object
  sensorObj = generate_sensor_object (privData, frame_meta);

  // analytics object
  analyticsObj = generate_analytics_module_object (privData, frame_meta);

  // object object
  objectObj = generate_object_object (privData, frame_meta, obj_meta);
  // event object
  eventObj = generate_event_object (obj_meta);

  char ts[MAX_TIME_STAMP_LEN + 1];
  generate_ts_rfc3339 (ts, MAX_TIME_STAMP_LEN);

  // root object
  rootObj = json_object_new ();
  json_object_set_string_member (rootObj, "messageid", msgIdStr);
  json_object_set_string_member (rootObj, "mdsversion", "1.0");
  json_object_set_string_member (rootObj, "@timestamp", ts);
  json_object_set_object_member (rootObj, "place", placeObj);
  json_object_set_object_member (rootObj, "sensor", sensorObj);
  json_object_set_object_member (rootObj, "analyticsModule", analyticsObj);
  json_object_set_object_member (rootObj, "object", objectObj);
  json_object_set_object_member (rootObj, "event", eventObj);

  json_object_set_string_member (rootObj, "videoPath", "");

  //Search for any custom message blob within frame usermeta list
  JsonArray *jArray = json_array_new ();
  for (NvDsUserMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
    NvDsUserMeta *frame_usermeta = (NvDsUserMeta *) l->data;
    if(frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
      NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *) frame_usermeta->user_meta_data;
      string msg = string((const char *) custom_blob->message, custom_blob->size);
      json_array_add_string_element (jArray, msg.c_str());
    }
  }
  if(json_array_get_length(jArray) > 0)
    json_object_set_array_member (rootObj, "customMessage", jArray);
  else
    json_array_unref(jArray);

  rootNode = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (rootNode, rootObj);

  message = json_to_string (rootNode, TRUE);
  json_node_free (rootNode);
  json_object_unref (rootObj);

  return message;

}

gchar* generate_dsmeta_message_minimal (void *privData, void *frameMeta)
{
  /*
  The JSON structure of the frame
  {
   "version": "4.0",
   "id": "frame-id",
   "@timestamp": "2018-04-11T04:59:59.828Z",
   "sensorId": "sensor-id",
   "objects": [
      ".......object-1 attributes...........",
      ".......object-2 attributes...........",
      ".......object-3 attributes..........."
    ]
  }
  */

  /*
  An example object with Vehicle object-type
  {
    "version": "4.0",
    "id": "frame-id",
    "@timestamp": "2018-04-11T04:59:59.828Z",
    "sensorId": "sensor-id",
    "objects": [
        "957|1834|150|1918|215|Vehicle|#|sedan|Bugatti|M|blue|CA 444|California|0.8",
        "..........."
    ]
  }
   */

  JsonNode *rootNode;
  JsonObject *jobject;
  JsonArray *jArray;
  stringstream ss;
  gchar *message = NULL;

  jArray = json_array_new ();

  NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) frameMeta;
  for (NvDsObjectMetaList *obj_l = frame_meta->obj_meta_list; obj_l; obj_l = obj_l->next) {
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) obj_l->data;
    if (obj_meta == NULL) {
      // Ignore Null object.
      continue;
    }

    // bbox sub object
    float scaleW = (float) frame_meta->source_frame_width /
                        (frame_meta->pipeline_width == 0) ? 1:frame_meta->pipeline_width;
    float scaleH = (float) frame_meta->source_frame_height /
                        (frame_meta->pipeline_height == 0) ? 1:frame_meta->pipeline_height;

    float left   = obj_meta->rect_params.left   * scaleW;
    float top    = obj_meta->rect_params.top    * scaleH;
    float width  = obj_meta->rect_params.width  * scaleW;
    float height = obj_meta->rect_params.height * scaleH;

    ss.str("");
    ss.clear();
    ss << obj_meta->object_id << "|" << left << "|" << top
       << "|" << left + width << "|" << top + height
       << "|" << obj_meta->obj_label;

    if(g_list_length(obj_meta->classifier_meta_list) > 0) {
        ss << "|#";
        //Add classifiers for the object, if any
        for(NvDsClassifierMetaList *cl = obj_meta->classifier_meta_list; cl ; cl=cl->next) {
          NvDsClassifierMeta *cl_meta = (NvDsClassifierMeta*) cl->data;
          for(NvDsLabelInfoList *ll = cl_meta->label_info_list; ll ; ll=ll->next) {
            NvDsLabelInfo *ll_meta = (NvDsLabelInfo*) ll->data;
            ss<< "|" << ll_meta->result_label;
          }
        }
        ss << "|" << obj_meta->confidence;
    }
    json_array_add_string_element (jArray, ss.str().c_str());
  }

  //generate timestamp
  char ts[MAX_TIME_STAMP_LEN + 1];
  generate_ts_rfc3339 (ts, MAX_TIME_STAMP_LEN);

  //fetch sensor id
  string sensorId="0";
  NvDsPayloadPriv *privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->sensorObj.find (frame_meta->source_id);
  if (idMap != privObj->sensorObj.end()) {
     NvDsSensorObject &obj = privObj->sensorObj[frame_meta->source_id];
     sensorId = obj.id;
  }

  jobject = json_object_new ();
  json_object_set_string_member (jobject, "version", "4.0");
  json_object_set_string_member (jobject, "id", to_string(frame_meta->frame_num).c_str());
  json_object_set_string_member (jobject, "@timestamp", ts);
  json_object_set_string_member (jobject, "sensorId", sensorId.c_str());

  json_object_set_array_member (jobject, "objects", jArray);

  JsonArray *custMsgjArray = json_array_new ();
  //Search for any custom message blob within frame usermeta list
  for (NvDsUserMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
    NvDsUserMeta *frame_usermeta = (NvDsUserMeta *) l->data;
    if(frame_usermeta && frame_usermeta->base_meta.meta_type == NVDS_CUSTOM_MSG_BLOB) {
      NvDsCustomMsgInfo *custom_blob = (NvDsCustomMsgInfo *) frame_usermeta->user_meta_data;
      string msg = string((const char *) custom_blob->message, custom_blob->size);
      json_array_add_string_element (custMsgjArray,  msg.c_str());
    }
  }
  if(json_array_get_length(custMsgjArray) > 0)
    json_object_set_array_member (jobject, "customMessage", custMsgjArray);
  else
    json_array_unref(custMsgjArray);

  rootNode = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (rootNode, jobject);

  message = json_to_string (rootNode, TRUE);
  json_node_free (rootNode);
  json_object_unref (jobject);

  return message;
}

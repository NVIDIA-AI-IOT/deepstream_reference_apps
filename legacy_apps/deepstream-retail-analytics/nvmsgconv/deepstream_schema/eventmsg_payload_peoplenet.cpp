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

static JsonObject*
generate_event_object (void *privData, NvDsEventMsgMeta *meta)
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

  switch (meta->type) {
    case NVDS_EVENT_ENTRY:
      json_object_set_string_member (eventObj, "type", "entry");
      break;
    case NVDS_EVENT_EXIT:
      json_object_set_string_member (eventObj, "type", "exit");
      break;
    case NVDS_EVENT_MOVING:
      json_object_set_string_member (eventObj, "type", "moving");
      break;
    case NVDS_EVENT_STOPPED:
      json_object_set_string_member (eventObj, "type", "stopped");
      break;
    case NVDS_EVENT_PARKED:
      json_object_set_string_member (eventObj, "type", "parked");
      break;
    case NVDS_EVENT_EMPTY:
      json_object_set_string_member (eventObj, "type", "empty");
      break;
    case NVDS_EVENT_RESET:
      json_object_set_string_member (eventObj, "type", "reset");
      break;
    default:
      cout << "Unknown event type " << endl;
      break;
  }

  return eventObj;
}

static JsonObject*
generate_object_object (void *privData, NvDsEventMsgMeta *meta)
{
  JsonObject *objectObj;
  JsonObject *jobject;
  guint i;
  gchar tracking_id[64];
  GList *objectMask = NULL;

  // object object
  objectObj = json_object_new ();
  if (snprintf (tracking_id, sizeof(tracking_id), "%lu", meta->trackingId)
      >= (int) sizeof(tracking_id))
    g_warning("Not enough space to copy trackingId");
  json_object_set_string_member (objectObj, "id", tracking_id);
  json_object_set_double_member (objectObj, "speed", 0);
  json_object_set_double_member (objectObj, "direction", 0);
  json_object_set_double_member (objectObj, "orientation", 0);

  switch (meta->objType) {
    case NVDS_OBJECT_TYPE_PERSON:
      // person sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsPersonObject *dsObj = (NvDsPersonObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_string_member (jobject, "hasBasket", dsObj->hasBasket);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No person object in meta data. Attach empty person sub object.
        json_object_set_string_member (jobject, "hasBasket", "NoBasket");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_string_member (objectObj, "detection", "person");
      json_object_set_object_member (objectObj, "obj_prop", jobject);
      break;
    case NVDS_OBJECT_TYPE_UNKNOWN:
      if(!meta->objectId) {
        break;
      }
      /** No information to add; object type unknown within NvDsEventMsgMeta */
      jobject = json_object_new ();
      json_object_set_object_member (objectObj, meta->objectId, jobject);
      break;
    default:
      cout << "Object type not implemented" << endl;
  }

  // bbox sub object
  jobject = json_object_new ();
  json_object_set_int_member (jobject, "topleftx", meta->bbox.left);
  json_object_set_int_member (jobject, "toplefty", meta->bbox.top);
  json_object_set_int_member (jobject, "bottomrightx", meta->bbox.left + meta->bbox.width);
  json_object_set_int_member (jobject, "bottomrighty", meta->bbox.top + meta->bbox.height);
  json_object_set_object_member (objectObj, "bbox", jobject);

  return objectObj;
}

gchar* generate_event_message (void *privData, NvDsEventMsgMeta *meta)
{
  JsonNode *rootNode;
  JsonObject *rootObj;
  JsonObject *eventObj;
  JsonObject *objectObj;
  gchar *message;

  uuid_t msgId;
  gchar msgIdStr[37];

  uuid_generate_random (msgId);
  uuid_unparse_lower(msgId, msgIdStr);

  // object object
  objectObj = generate_object_object (privData, meta);

  // event object
  eventObj = generate_event_object (privData, meta);

  // root object
  rootObj = json_object_new ();
  json_object_set_string_member (rootObj, "messageid", msgIdStr);
  json_object_set_string_member (rootObj, "mdsversion", "1.0");
  json_object_set_string_member (rootObj, "timestamp", meta->ts);
  json_object_set_object_member (rootObj, "object", objectObj);
  json_object_set_object_member (rootObj, "event_des", eventObj);

  if (meta->videoPath)
    json_object_set_string_member (rootObj, "videoPath", meta->videoPath);
  else
    json_object_set_string_member (rootObj, "videoPath", "");

  rootNode = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (rootNode, rootObj);

  message = json_to_string (rootNode, TRUE);
  json_node_free (rootNode);
  json_object_unref (rootObj);

  return message;
}

static const gchar*
object_enum_to_str (NvDsObjectType type, gchar* objectId)
{
  switch (type) {
    case NVDS_OBJECT_TYPE_VEHICLE:
      return "Vehicle";
    case NVDS_OBJECT_TYPE_FACE:
      return "Face";
    case NVDS_OBJECT_TYPE_PERSON:
      return "Person";
    case NVDS_OBJECT_TYPE_BAG:
      return "Bag";
    case NVDS_OBJECT_TYPE_BICYCLE:
      return "Bicycle";
    case NVDS_OBJECT_TYPE_ROADSIGN:
      return "RoadSign";
    case NVDS_OBJECT_TYPE_CUSTOM:
      return "Custom";
    case NVDS_OBJECT_TYPE_UNKNOWN:
      return objectId ? objectId : "Unknown";
    default:
      return "Unknown";
  }
}

static const gchar*
to_str (gchar* cstr)
{
    return reinterpret_cast<const gchar*>(cstr) ? cstr : "";
}

static const gchar *
sensor_id_to_str (void *privData, gint sensorId)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject *dsObj = NULL;

  g_return_val_if_fail (privData, NULL);

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->sensorObj.find (sensorId);
  if (idMap != privObj->sensorObj.end()) {
    dsObj = &idMap->second;
    return dsObj->id.c_str();
  } else {
    cout << "No entry for " CONFIG_GROUP_SENSOR << sensorId
        << " in configuration file" << endl;
    return NULL;
  }
}

static void
generate_mask_array (NvDsEventMsgMeta *meta, JsonArray *jArray, GList *mask)
{
  unsigned int i;
  GList *l;
  stringstream ss;
  bool started = false;

  ss << meta->trackingId << "|" << g_list_length(mask);

  for (l = mask; l != NULL; l = l->next) {
    GArray *polygon = (GArray *) l->data;

    if (started)
      ss << "|#";

    started = true;

    for (i = 0; i < polygon->len; i++) {
      gdouble value = g_array_index (polygon, gdouble, i);
      ss << "|" << value;
    }
  }
  json_array_add_string_element (jArray, ss.str().c_str());
}

gchar* generate_event_message_minimal (void *privData, NvDsEvent *events, guint size)
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
  JsonArray *maskArray = NULL;
  guint i;
  stringstream ss;
  gchar *message = NULL;

  jArray = json_array_new ();

  for (i = 0; i < size; i++) {
    GList *objectMask = NULL;

    ss.str("");
    ss.clear();

    NvDsEventMsgMeta *meta = events[i].metadata;
    ss << meta->trackingId << "|" << meta->bbox.left << "|" << meta->bbox.top
        << "|" << meta->bbox.left + meta->bbox.width << "|" << meta->bbox.top + meta->bbox.height
        << "|" << object_enum_to_str (meta->objType, meta->objectId);

    if (meta->extMsg && meta->extMsgSize) {
      // Attach secondary inference attributes.
      switch (meta->objType) {
        case NVDS_OBJECT_TYPE_VEHICLE: {
          NvDsVehicleObject *dsObj = (NvDsVehicleObject *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->type) << "|" << to_str(dsObj->make) << "|"
               << to_str(dsObj->model) << "|" << to_str(dsObj->color) << "|" << to_str(dsObj->license)
               << "|" << to_str(dsObj->region) << "|" << meta->confidence;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_PERSON: {
          NvDsPersonObject *dsObj = (NvDsPersonObject *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->apparel)
                << "|" << meta->confidence;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_FACE: {
          NvDsFaceObject *dsObj = (NvDsFaceObject *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->glasses)
                << "|" << to_str(dsObj->facialhair) << "|" << to_str(dsObj->name) << "|"
                << "|" << to_str(dsObj->eyecolor) << "|" << meta->confidence;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_VEHICLE_EXT: {
          NvDsVehicleObjectExt *dsObj = (NvDsVehicleObjectExt *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->type) << "|" << to_str(dsObj->make) << "|"
               << to_str(dsObj->model) << "|" << to_str(dsObj->color) << "|" << to_str(dsObj->license)
               << "|" << to_str(dsObj->region) << "|" << meta->confidence;

            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_PERSON_EXT: {
          NvDsPersonObjectExt *dsObj = (NvDsPersonObjectExt *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->apparel)
                << "|" << meta->confidence;

            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        case NVDS_OBJECT_TYPE_FACE_EXT: {
          NvDsFaceObjectExt *dsObj = (NvDsFaceObjectExt *) meta->extMsg;
          if (dsObj) {
            ss << "|#|" << to_str(dsObj->gender) << "|" << dsObj->age << "|"
                << to_str(dsObj->hair) << "|" << to_str(dsObj->cap) << "|" << to_str(dsObj->glasses)
                << "|" << to_str(dsObj->facialhair) << "|" << to_str(dsObj->name) << "|"
                << "|" << to_str(dsObj->eyecolor) << "|" << meta->confidence;

            if (dsObj->mask)
              objectMask = dsObj->mask;
          }
        }
          break;
        default:
          cout << "Object type (" << meta->objType << ") not implemented" << endl;
          break;
      }
    }

    if (objectMask) {
      if (maskArray == NULL)
        maskArray = json_array_new ();
      generate_mask_array (meta, maskArray, objectMask);
    }

    json_array_add_string_element (jArray, ss.str().c_str());
  }

  // It is assumed that all events / objects are associated with same frame.
  // Therefore ts / sensorId / frameId of first object can be used.

  jobject = json_object_new ();
  json_object_set_string_member (jobject, "version", "4.0");
  json_object_set_string_member (jobject, "id", to_string(events[0].metadata->frameId).c_str());
  json_object_set_string_member (jobject, "@timestamp", events[0].metadata->ts);
  if (events[0].metadata->sensorStr) {
    json_object_set_string_member (jobject, "sensorId", events[0].metadata->sensorStr);
  } else if ((NvDsPayloadPriv *) privData) {
    json_object_set_string_member (jobject, "sensorId",
        to_str((gchar *) sensor_id_to_str (privData, events[0].metadata->sensorId)));
  } else {
    json_object_set_string_member (jobject, "sensorId", "0");
  }

  json_object_set_array_member (jobject, "objects", jArray);
  if (maskArray && json_array_get_length (maskArray) > 0)
    json_object_set_array_member (jobject, "masks", maskArray);

  rootNode = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (rootNode, jobject);

  message = json_to_string (rootNode, TRUE);
  json_node_free (rootNode);
  json_object_unref (jobject);

  return message;
}

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "deepstream_schema.h"

constexpr int NVDS_OBJECT_TYPE_PERSON_EXT_POSE = 0x103; //NVDS_OBJECT_TYPE_UNKNOWN = 102
static std::vector<std::vector<std::string>> _joint_maps = {
  {// Type 0 - pose2D
    std::string("nose"),
    std::string("neck"),
    std::string("right-shoulder"),
    std::string("right-elbow"),
    std::string("right-hand"),
    std::string("left-shoulder"),
    std::string("left-elbow"),
    std::string("left-hand"),
    std::string("right-hip"),
    std::string("right-knee"),
    std::string("right-foot"),
    std::string("left-hip"),
    std::string("left-knee"),
    std::string("left-foot"),
    std::string("right-eye"),
    std::string("left-eye"),
    std::string("right-ear"),
    std::string("left-ear"),
  },
  {
    std::string("pelvis"),
    std::string("left-hip"),
    std::string("right-hip"),
    std::string("torso"),
    std::string("left-knee"),
    std::string("right-knee"),
    std::string("neck"),
    std::string("left-ankle"),
    std::string("right-ankle"),
    std::string("left-big-toe"),
    std::string("right-big-toe"),
    std::string("left-small-toe"),
    std::string("right-small-toe"),
    std::string("left-heel"),
    std::string("right-heel"),
    std::string("nose"),
    std::string("left-eye"),
    std::string("right-eye"),
    std::string("left-ear"),
    std::string("right-ear"),
    std::string("left-shoulder"),
    std::string("right-shoulder"),
    std::string("left-elbow"),
    std::string("right-elbow"),
    std::string("left-wrist"),
    std::string("right-wrist"),
    std::string("left-pinky-knuckle"),
    std::string("right-pinky-knuckle"),
    std::string("left-middle-tip"),
    std::string("right-middle-tip"),
    std::string("left-index-knuckle"),
    std::string("right-index-knuckle"),
    std::string("left-thumb-tip"),
    std::string("right-thumb-tip"),
  },
  {
    std::string("pelvis"),
    std::string("left-hip"),
    std::string("right-hip"),
    std::string("torso"),
    std::string("left-knee"),
    std::string("right-knee"),
    std::string("neck"),
    std::string("left-ankle"),
    std::string("right-ankle"),
    std::string("left-big-toe"),
    std::string("right-big-toe"),
    std::string("left-small-toe"),
    std::string("right-small-toe"),
    std::string("left-heel"),
    std::string("right-heel"),
    std::string("nose"),
    std::string("left-eye"),
    std::string("right-eye"),
    std::string("left-ear"),
    std::string("right-ear"),
    std::string("left-shoulder"),
    std::string("right-shoulder"),
    std::string("left-elbow"),
    std::string("right-elbow"),
    std::string("left-wrist"),
    std::string("right-wrist"),
    std::string("left-pinky-knuckle"),
    std::string("right-pinky-knuckle"),
    std::string("left-middle-tip"),
    std::string("right-middle-tip"),
    std::string("left-index-knuckle"),
    std::string("right-index-knuckle"),
    std::string("left-thumb-tip"),
    std::string("right-thumb-tip"),
  }
};

typedef struct NvDsJoint {
  gdouble confidence;
  gdouble x;
  gdouble y;
  gdouble z;
}NvDsJoint;

typedef struct NvDsJoints {

  gint pose_type;
  gint num_joints;
  NvDsJoint *joints;

}NvDsJoints;

typedef struct NvDsPersonPoseExt {
  gint num_poses;
  NvDsJoints *poses;
}NvDsPersonPoseExt;

static JsonObject*
generate_place_object (void *privData, NvDsEventMsgMeta *meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsPlaceObject *dsPlaceObj = NULL;
  JsonObject *placeObj;
  JsonObject *jobject;
  JsonObject *jobject2;

  privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->placeObj.find (meta->placeId);

  if (idMap != privObj->placeObj.end()) {
    dsPlaceObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_PLACE << meta->placeId
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

  // parkingSpot / aisle /entrance sub object
  jobject = json_object_new ();

  switch (meta->type) {
    case NVDS_EVENT_MOVING:
    case NVDS_EVENT_STOPPED:
      json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
      json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field2.c_str());
      json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
      json_object_set_object_member (placeObj, "aisle", jobject);
      break;
    case NVDS_EVENT_EMPTY:
    case NVDS_EVENT_PARKED:
      json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
      json_object_set_string_member (jobject, "type", dsPlaceObj->subObj.field2.c_str());
      json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
      json_object_set_object_member (placeObj, "parkingSpot", jobject);
      break;
    case NVDS_EVENT_ENTRY:
    case NVDS_EVENT_EXIT:
      if (meta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
        json_object_set_string_member (jobject, "id", dsPlaceObj->subObj.field1.c_str());
        json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field2.c_str());
        json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
        json_object_set_object_member (placeObj, "aisle", jobject);
      } else {
        json_object_set_string_member (jobject, "name", dsPlaceObj->subObj.field1.c_str());
        json_object_set_string_member (jobject, "lane", dsPlaceObj->subObj.field2.c_str());
        json_object_set_string_member (jobject, "level", dsPlaceObj->subObj.field3.c_str());
        json_object_set_object_member (placeObj, "entrance", jobject);
      }
      break;
    default:
      cout << "Event type not implemented " << endl;
      break;
  }

  // coordinate sub sub object
  jobject2 = json_object_new ();
  json_object_set_double_member (jobject2, "x", dsPlaceObj->coordinate[0]);
  json_object_set_double_member (jobject2, "y", dsPlaceObj->coordinate[1]);
  json_object_set_double_member (jobject2, "z", dsPlaceObj->coordinate[2]);
  json_object_set_object_member (jobject, "coordinate", jobject2);

  return placeObj;
}

static JsonObject*
generate_sensor_object (void *privData, NvDsEventMsgMeta *meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsSensorObject *dsSensorObj = NULL;
  JsonObject *sensorObj;
  JsonObject *jobject;

  privObj = (NvDsPayloadPriv *) privData;
  auto idMap = privObj->sensorObj.find (meta->sensorId);

  if (idMap != privObj->sensorObj.end()) {
    dsSensorObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_SENSOR << meta->sensorId
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

static JsonObject*
generate_analytics_module_object (void *privData, NvDsEventMsgMeta *meta)
{
  NvDsPayloadPriv *privObj = NULL;
  NvDsAnalyticsObject *dsObj = NULL;
  JsonObject *analyticsObj;

  privObj = (NvDsPayloadPriv *) privData;

  auto idMap = privObj->analyticsObj.find (meta->moduleId);

  if (idMap != privObj->analyticsObj.end()) {
    dsObj = &idMap->second;
  } else {
    cout << "No entry for " CONFIG_GROUP_ANALYTICS << meta->moduleId
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
  JsonObject *pobject;
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
    case NVDS_OBJECT_TYPE_VEHICLE:
      // vehicle sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsVehicleObject *dsObj = (NvDsVehicleObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_string_member (jobject, "type", dsObj->type);
          json_object_set_string_member (jobject, "make", dsObj->make);
          json_object_set_string_member (jobject, "model", dsObj->model);
          json_object_set_string_member (jobject, "color", dsObj->color);
          json_object_set_string_member (jobject, "licenseState", dsObj->region);
          json_object_set_string_member (jobject, "license", dsObj->license);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No vehicle object in meta data. Attach empty vehicle sub object.
        json_object_set_string_member (jobject, "type", "");
        json_object_set_string_member (jobject, "make", "");
        json_object_set_string_member (jobject, "model", "");
        json_object_set_string_member (jobject, "color", "");
        json_object_set_string_member (jobject, "licenseState", "");
        json_object_set_string_member (jobject, "license", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "vehicle", jobject);
      break;
    case NVDS_OBJECT_TYPE_PERSON:
      // person sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsPersonObject *dsObj = (NvDsPersonObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "apparel", dsObj->apparel);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No person object in meta data. Attach empty person sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "apparel", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "person", jobject);
      break;
    case NVDS_OBJECT_TYPE_FACE:
      // face sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsFaceObject *dsObj = (NvDsFaceObject *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "glasses", dsObj->glasses);
          json_object_set_string_member (jobject, "facialhair", dsObj->facialhair);
          json_object_set_string_member (jobject, "name", dsObj->name);
          json_object_set_string_member (jobject, "eyecolor", dsObj->eyecolor);
          json_object_set_double_member (jobject, "confidence", meta->confidence);
        }
      } else {
        // No face object in meta data. Attach empty face sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "glasses", "");
        json_object_set_string_member (jobject, "facialhair", "");
        json_object_set_string_member (jobject, "name", "");
        json_object_set_string_member (jobject, "eyecolor", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "face", jobject);
      break;
    case NVDS_OBJECT_TYPE_VEHICLE_EXT:
      // vehicle sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsVehicleObjectExt *dsObj = (NvDsVehicleObjectExt *) meta->extMsg;
        if (dsObj) {
          json_object_set_string_member (jobject, "type", dsObj->type);
          json_object_set_string_member (jobject, "make", dsObj->make);
          json_object_set_string_member (jobject, "model", dsObj->model);
          json_object_set_string_member (jobject, "color", dsObj->color);
          json_object_set_string_member (jobject, "licenseState", dsObj->region);
          json_object_set_string_member (jobject, "license", dsObj->license);
          json_object_set_double_member (jobject, "confidence", meta->confidence);

          objectMask = dsObj->mask;
        }
      } else {
        // No vehicle object in meta data. Attach empty vehicle sub object.
        json_object_set_string_member (jobject, "type", "");
        json_object_set_string_member (jobject, "make", "");
        json_object_set_string_member (jobject, "model", "");
        json_object_set_string_member (jobject, "color", "");
        json_object_set_string_member (jobject, "licenseState", "");
        json_object_set_string_member (jobject, "license", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "vehicle", jobject);
      break;
    case NVDS_OBJECT_TYPE_PERSON_EXT:
      // person sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsPersonObjectExt *dsObj = (NvDsPersonObjectExt *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "apparel", dsObj->apparel);
          json_object_set_double_member (jobject, "confidence", meta->confidence);

          objectMask = dsObj->mask;
        }
      } else {
        // No person object in meta data. Attach empty person sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "apparel", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "person", jobject);
      break;
    case NVDS_OBJECT_TYPE_FACE_EXT:
      // face sub object
      jobject = json_object_new ();

      if (meta->extMsgSize) {
        NvDsFaceObjectExt *dsObj = (NvDsFaceObjectExt *) meta->extMsg;
        if (dsObj) {
          json_object_set_int_member (jobject, "age", dsObj->age);
          json_object_set_string_member (jobject, "gender", dsObj->gender);
          json_object_set_string_member (jobject, "hair", dsObj->hair);
          json_object_set_string_member (jobject, "cap", dsObj->cap);
          json_object_set_string_member (jobject, "glasses", dsObj->glasses);
          json_object_set_string_member (jobject, "facialhair", dsObj->facialhair);
          json_object_set_string_member (jobject, "name", dsObj->name);
          json_object_set_string_member (jobject, "eyecolor", dsObj->eyecolor);
          json_object_set_double_member (jobject, "confidence", meta->confidence);

          objectMask = dsObj->mask;
        }
      } else {
        // No face object in meta data. Attach empty face sub object.
        json_object_set_int_member (jobject, "age", 0);
        json_object_set_string_member (jobject, "gender", "");
        json_object_set_string_member (jobject, "hair", "");
        json_object_set_string_member (jobject, "cap", "");
        json_object_set_string_member (jobject, "glasses", "");
        json_object_set_string_member (jobject, "facialhair", "");
        json_object_set_string_member (jobject, "name", "");
        json_object_set_string_member (jobject, "eyecolor", "");
        json_object_set_double_member (jobject, "confidence", 1.0);
      }
      json_object_set_object_member (objectObj, "face", jobject);
      break;
    case NVDS_OBJECT_TYPE_UNKNOWN:
      if(!meta->objectId) {
        break;
      }
      /** No information to add; object type unknown within NvDsEventMsgMeta */
      jobject = json_object_new ();
      json_object_set_object_member (objectObj, meta->objectId, jobject);
      break;
    case (NvDsObjectType)NVDS_OBJECT_TYPE_PERSON_EXT_POSE:
      {
        if (meta->extMsgSize){
          NvDsPersonPoseExt *pose_meta = (NvDsPersonPoseExt*)meta->extMsg;

          // // DEBUG
          // g_message("generate_object_object(): pose_meta->num_poses = %d", pose_meta->num_poses);

          for (int i=0; i<pose_meta->num_poses; i++) {
            // bbox sub object
            jobject = json_object_new ();
            json_object_set_int_member (jobject, "topleftx", meta->bbox.left);
            json_object_set_int_member (jobject, "toplefty", meta->bbox.top);
            json_object_set_int_member (jobject, "bottomrightx", meta->bbox.left + meta->bbox.width);
            json_object_set_int_member (jobject, "bottomrighty", meta->bbox.top + meta->bbox.height);
            json_object_set_object_member (objectObj, "bbox", jobject);

            pobject = json_object_new ();
            int joint_index = 0;

            for (joint_index = 0;joint_index < pose_meta->poses[i].num_joints; joint_index++)
            {
              if (pose_meta->poses[i].joints[joint_index].confidence > 0.0)
              {
                jobject = json_object_new ();
                std::string s = _joint_maps[pose_meta->poses[i].pose_type][joint_index];
                char *json_name = const_cast<char*>(s.c_str());

                if (pose_meta->poses[i].pose_type==0) {
                  json_object_set_int_member (jobject, "x", pose_meta->poses[i].joints[joint_index].x);
                  json_object_set_int_member (jobject, "y", pose_meta->poses[i].joints[joint_index].y);
                  json_object_set_int_member (jobject, "confidence", pose_meta->poses[i].joints[joint_index].confidence);
                  json_object_set_object_member (pobject, reinterpret_cast<const gchar*>(json_name), jobject);
                }
                else if ((pose_meta->poses[i].pose_type==1)||(pose_meta->poses[i].pose_type==2)) {
                  // pose3d or pose25d from BodyPose3DNet
                  json_object_set_int_member (jobject, "x", pose_meta->poses[i].joints[joint_index].x);
                  json_object_set_int_member (jobject, "y", pose_meta->poses[i].joints[joint_index].y);
                  json_object_set_int_member (jobject, "z", pose_meta->poses[i].joints[joint_index].z);
                  json_object_set_int_member (jobject, "confidence", pose_meta->poses[i].joints[joint_index].confidence);
                  json_object_set_object_member (pobject, reinterpret_cast<const gchar*>(json_name), jobject);
                }

              }
            }

            if      (pose_meta->poses[i].pose_type==0)
              json_object_set_object_member (objectObj, "pose2D", pobject);
            else if (pose_meta->poses[i].pose_type==1)
              json_object_set_object_member (objectObj, "pose3D", pobject);
            else if (pose_meta->poses[i].pose_type==2)
              json_object_set_object_member (objectObj, "pose25D", pobject);
          }
        }
      }
      break;
    default:
      cout << "Object type not implemented" << endl;
      break;
  }


  if (objectMask) {
    GList *l;
    JsonArray *maskArray = json_array_sized_new (g_list_length(objectMask));

    for (l = objectMask; l != NULL; l = l->next) {
      GArray *polygon = (GArray *) l->data;
      JsonArray *polygonArray = json_array_sized_new (polygon->len);

      for (i = 0; i < polygon->len; i++) {
        gdouble value = g_array_index (polygon, gdouble, i);

        json_array_add_double_element (polygonArray, value);
      }

      json_array_add_array_element (maskArray, polygonArray);
    }

    json_object_set_array_member (objectObj, "maskoutline", maskArray);
  }

  // signature sub array
  if (meta->objSignature.size) {
    JsonArray *jArray = json_array_sized_new (meta->objSignature.size);

    for (i = 0; i < meta->objSignature.size; i++) {
      json_array_add_double_element (jArray, meta->objSignature.signature[i]);
    }
    json_object_set_array_member (objectObj, "signature", jArray);
  }

  // location sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "lat", meta->location.lat);
  json_object_set_double_member (jobject, "lon", meta->location.lon);
  json_object_set_double_member (jobject, "alt", meta->location.alt);
  json_object_set_object_member (objectObj, "location", jobject);

  // coordinate sub object
  jobject = json_object_new ();
  json_object_set_double_member (jobject, "x", meta->coordinate.x);
  json_object_set_double_member (jobject, "y", meta->coordinate.y);
  json_object_set_double_member (jobject, "z", meta->coordinate.z);
  json_object_set_object_member (objectObj, "coordinate", jobject);

  return objectObj;
}

gchar* generate_event_message (void *privData, NvDsEventMsgMeta *meta)
{
  JsonNode *rootNode;
  JsonObject *rootObj;
  JsonObject *placeObj;
  JsonObject *sensorObj;
  JsonObject *analyticsObj;
  JsonObject *eventObj;
  JsonObject *objectObj;
  gchar *message;

  uuid_t msgId;
  gchar msgIdStr[37];

  uuid_generate_random (msgId);
  uuid_unparse_lower(msgId, msgIdStr);

  // place object
  placeObj = generate_place_object (privData, meta);

  // sensor object
  sensorObj = generate_sensor_object (privData, meta);

  // analytics object
  analyticsObj = generate_analytics_module_object (privData, meta);

  // object object
  objectObj = generate_object_object (privData, meta);

  // event object
  eventObj = generate_event_object (privData, meta);

  // root object
  rootObj = json_object_new ();
  json_object_set_string_member (rootObj, "messageid", msgIdStr);
  json_object_set_string_member (rootObj, "mdsversion", "1.0");
  json_object_set_string_member (rootObj, "@timestamp", meta->ts);
  json_object_set_object_member (rootObj, "place", placeObj);
  json_object_set_object_member (rootObj, "sensor", sensorObj);
  json_object_set_object_member (rootObj, "analyticsModule", analyticsObj);
  json_object_set_object_member (rootObj, "object", objectObj);
  json_object_set_object_member (rootObj, "event", eventObj);

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
    case (NvDsObjectType)NVDS_OBJECT_TYPE_PERSON_EXT_POSE:
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
        case (NvDsObjectType)NVDS_OBJECT_TYPE_PERSON_EXT_POSE: {
          NvDsPersonPoseExt *pose_meta = (NvDsPersonPoseExt *) meta->extMsg;
          if (pose_meta) {
            // STUB person
            ss << "|#||||||" << meta->confidence;

            for (int i = 0; i < pose_meta->num_poses; i++) {
              int type = pose_meta->poses[i].pose_type;

              if      (type == 0) {
                ss << "|#|pose2D|";
                for (int joint_index = 0; joint_index < pose_meta->poses[i].num_joints; joint_index++) {
                    std::string s = _joint_maps[type][joint_index]
                                      + "," + std::to_string(pose_meta->poses[i].joints[joint_index].x)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].y)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].confidence);
                    ss << s << "|";
                }
              }
              else if (type == 1) {// BodyPose3DNet
                ss << "|#|pose3D|";
                for (int joint_index = 0; joint_index < pose_meta->poses[i].num_joints; joint_index++) {
                    std::string s = _joint_maps[type][joint_index]
                                      + "," + std::to_string(pose_meta->poses[i].joints[joint_index].x)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].y)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].z)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].confidence);
                    ss << s << "|";
                }
              }
              else if (type == 2) {// BodyPose3DNet
                ss << "|#|pose25D|";
                for (int joint_index = 0; joint_index < pose_meta->poses[i].num_joints; joint_index++) {
                    std::string s = _joint_maps[type][joint_index]
                                      + "," + std::to_string(pose_meta->poses[i].joints[joint_index].x)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].y)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].z)
                                      + "," + to_string(pose_meta->poses[i].joints[joint_index].confidence);
                    ss << s << "|";
                }
              }
            }
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
  json_object_set_int_member (jobject, "id", events[0].metadata->frameId);
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


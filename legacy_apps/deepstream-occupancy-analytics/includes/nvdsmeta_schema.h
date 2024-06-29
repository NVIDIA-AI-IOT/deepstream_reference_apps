/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

/**
 * @file
 * <b>NVIDIA DeepStream: Metadata Extension Structures</b>
 *
 * @b Description: This file defines the NVIDIA DeepStream metadata structures
 * used to describe metadata objects.
 */

/**
 * @defgroup  metadata_extensions  Metadata Extension Structures
 *
 * Defines metadata structures used to describe metadata objects.
 *
 * @ingroup NvDsMetaApi
 * @{
 */

#ifndef NVDSMETA_H_
#define NVDSMETA_H_

#include <glib.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Defines event type flags.
 */
typedef enum NvDsEventType {
  NVDS_EVENT_ENTRY,
  NVDS_EVENT_EXIT,
  NVDS_EVENT_MOVING,
  NVDS_EVENT_STOPPED,
  NVDS_EVENT_EMPTY,
  NVDS_EVENT_PARKED,
  NVDS_EVENT_RESET,

  /** Reserved for future use. Custom events must be assigned values
   greater than this. */
  NVDS_EVENT_RESERVED = 0x100,
  /** Specifies a custom event. */
  NVDS_EVENT_CUSTOM = 0x101,
  NVDS_EVENT_FORCE32 = 0x7FFFFFFF
} NvDsEventType;

/**
 * Defines object type flags.
 */
typedef enum NvDsObjectType {
  NVDS_OBJECT_TYPE_VEHICLE,
  NVDS_OBJECT_TYPE_PERSON,
  NVDS_OBJECT_TYPE_FACE,
  NVDS_OBJECT_TYPE_BAG,
  NVDS_OBJECT_TYPE_BICYCLE,
  NVDS_OBJECT_TYPE_ROADSIGN,
  /** Reserved for future use. Custom objects must be assigned values
   greater than this. */
  NVDS_OBJECT_TYPE_RESERVED = 0x100,
  /** Specifies a custom object. */
  NVDS_OBJECT_TYPE_CUSTOM = 0x101,
  /** "object" key will be missing in the schema */
  NVDS_OBJECT_TYPE_UNKNOWN = 0x102,
  NVDS_OBEJCT_TYPE_FORCE32 = 0x7FFFFFFF
} NvDsObjectType;

/**
 * Defines payload type flags.
 */
typedef enum NvDsPayloadType {
  NVDS_PAYLOAD_DEEPSTREAM,
  NVDS_PAYLOAD_DEEPSTREAM_MINIMAL,
  /** Reserved for future use. Custom payloads must be assigned values
   greater than this. */
  NVDS_PAYLOAD_RESERVED = 0x100,
  /** Specifies a custom payload. You must implement the nvds_msg2p_*
   interface. */
  NVDS_PAYLOAD_CUSTOM = 0x101,
  NVDS_PAYLOAD_FORCE32 = 0x7FFFFFFF
} NvDsPayloadType;

/**
 * Holds a rectangle's position and size.
 */
typedef struct NvDsRect {
  float top;     /**< Holds the position of rectangle's top in pixels. */
  float left;    /**< Holds the position of rectangle's left side in pixels. */
  float width;   /**< Holds the rectangle's width in pixels. */
  float height;  /**< Holds the rectangle's height in pixels. */
} NvDsRect;

/**
 * Holds geolocation parameters.
 */
typedef struct NvDsGeoLocation {
  gdouble lat;      /**< Holds the location's latitude. */
  gdouble lon;      /**< Holds the location's longitude. */
  gdouble alt;      /**< Holds the location's altitude. */
} NvDsGeoLocation;

/**
 * Hold a coordinate's position.
 */
typedef struct NvDsCoordinate {
  gdouble x;        /**< Holds the coordinate's X position. */
  gdouble y;        /**< Holds the coordinate's Y position. */
  gdouble z;        /**< Holds the coordinate's Z position. */
} NvDsCoordinate;

/**
 * Holds an object's signature.
 */
typedef struct NvDsObjectSignature {
  /** Holds a pointer to an array of signature values. */
  gdouble *signature;
  /** Holds the number of signature values in @a signature. */
  guint size;
} NvDsObjectSignature;

/**
 * Holds a vehicle object's parameters.
 */
typedef struct NvDsVehicleObject {
  gchar *type;      /**< Holds a pointer to the type of the vehicle. */
  gchar *make;      /**< Holds a pointer to the make of the vehicle. */
  gchar *model;     /**< Holds a pointer to the model of the vehicle. */
  gchar *color;     /**< Holds a pointer to the color of the vehicle. */
  gchar *region;    /**< Holds a pointer to the region of the vehicle. */
  gchar *license;   /**< Holds a pointer to the license number of the vehicle.*/
} NvDsVehicleObject;

/**
 * Holds a person object's parameters.
 */
typedef struct NvDsPersonObject {
  gchar *gender;    /**< Holds a pointer to the person's gender. */
  gchar *hair;      /**< Holds a pointer to the person's hair color. */
  gchar *cap;       /**< Holds a pointer to the type of cap the person is
                     wearing, if any. */
  gchar *apparel;   /**< Holds a pointer to a description of the person's
                     apparel. */
  guint age;        /**< Holds the person's age. */
} NvDsPersonObject;

/**
 * Holds a face object's parameters.
 */
typedef struct NvDsFaceObject {
  gchar *gender;    /**< Holds a pointer to the person's gender. */
  gchar *hair;      /**< Holds a pointer to the person's hair color. */
  gchar *cap;       /**< Holds a pointer to the type of cap the person
                     is wearing, if any. */
  gchar *glasses;   /**< Holds a pointer to the type of glasses the person
                     is wearing, if any. */
  gchar *facialhair;/**< Holds a pointer to the person's facial hair color. */
  gchar *name;      /**< Holds a pointer to the person's name. */
  gchar *eyecolor;  /**< Holds a pointer to the person's eye color. */
  guint age;        /**< Holds the person's age. */
} NvDsFaceObject;

/**
 * Holds event message meta data.
 *
 * You can attach various types of objects (vehicle, person, face, etc.)
 * to an event by setting a pointer to the object in @a extMsg.
 *
 * Similarly, you can attach a custom object to an event by setting a pointer to the object in @a extMsg.
 * A custom object must be handled by the metadata parsing module accordingly.
 */
typedef struct NvDsEventMsgMeta {
  /** Holds the event's type. */
  NvDsEventType type;
  /** Holds the object's type. */
  NvDsObjectType objType;
  /** Holds the object's bounding box. */
  NvDsRect bbox;
  /** Holds the object's geolocation. */
  NvDsGeoLocation location;
  /** Holds the object's coordinates. */
  NvDsCoordinate coordinate;
  /** Holds the object's signature. */
  NvDsObjectSignature objSignature;
  /** Holds the object's class ID. */
  gint objClassId;
  /** Holds the ID of the sensor that generated the event. */
  gint sensorId;
  /** Holds the ID of the analytics module that generated the event. */
  gint moduleId;
  /** Holds the ID of the place related to the object. */
  gint placeId;
  /** Holds the ID of the component (plugin) that generated this event. */
  gint componentId;
  /** Holds the video frame ID of this event. */
  gint frameId;
  /** Holds the confidence level of the inference. */
  gdouble confidence;
  /** Holds the object's tracking ID. */
  gint trackingId;
  /** Holds a pointer to the generated event's timestamp. */
  gchar *ts;
  /** Holds a pointer to the detected or inferred object's ID. */
  gchar *objectId;

  /** Holds a pointer to a string containing the sensor's identity. */
  gchar *sensorStr;
  /** Holds a pointer to a string containing other attributes associated with
   the object. */
  gchar *otherAttrs;
  /** Holds a pointer to the name of the video file. */
  gchar *videoPath;
  /** Holds a pointer to event message meta data. This can be used to hold
   data that can't be accommodated in the existing fields, or an associated
   object (representing a vehicle, person, face, etc.). */
  gpointer extMsg;
  /** Holds the size of the custom object at @a extMsg. */
  guint extMsgSize;
  
  /*My data*/
  guint occupancy;
  guint source_id;
  guint lccum_cnt_entry;
  guint lccum_cnt_exit;
} NvDsEventMsgMeta;

/**
 * Holds event information.
 */
typedef struct _NvDsEvent {
  /** Holds the type of event. */
  NvDsEventType eventType;
  /** Holds a pointer to event metadata. */
  NvDsEventMsgMeta *metadata;
} NvDsEvent;

/**
 * Holds payload metadata.
 */
typedef struct NvDsPayload {
  /** Holds a pointer to the payload. */
  gpointer payload;
  /** Holds the size of the payload. */
  guint payloadSize;
  /** Holds the ID of the component (plugin) which attached the payload
   (optional). */
  guint componentId;
} NvDsPayload;

#ifdef __cplusplus
}
#endif
#endif /* NVDSMETA_H_ */

/** @} */

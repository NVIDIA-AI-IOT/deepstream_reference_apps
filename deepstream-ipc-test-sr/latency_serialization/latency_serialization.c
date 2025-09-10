/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>

void serialize_meta(GstBuffer *buf, guint8 **data, guint *len) {
  if (buf == NULL || data == NULL || len == NULL) {
    g_print("Invalid arguments\n");
    return;
  }
  guint out_len = 0;
  GstReferenceTimestampMeta *meta =
      gst_buffer_get_reference_timestamp_meta(buf, NULL);
  if (meta == NULL) {
    // g_print("serialize_meta: no reference timestamp meta\n");
    return;
  }
  GstCaps *ref = meta->reference;
  if (ref) {
    gchar *caps_str = gst_caps_to_string (ref);
    out_len = strlen(caps_str) + 1;
    // g_print("caps_str %s\n", caps_str);
    *len = out_len;
    // Allocate memory for the serialized data, free it after use
    *data = g_malloc0(*len);
    memcpy(*data, caps_str, out_len);
  } else {
    *data = NULL;
    *len = 0;
  }
}

void deserialize_meta(GstBuffer *buf, guint8 *data, guint len) {
  if (buf == NULL || data == NULL || len == 0) {
    g_print("Invalid arguments\n");
    return;
  }
  GstCaps *caps = gst_caps_from_string((const gchar *)data);
  if (caps) {
    gst_buffer_add_reference_timestamp_meta(buf, caps, 0, 0);
    gst_caps_unref(caps);
  }
  // g_print("deserialize_meta data: %s\n", (const gchar *)data);
}

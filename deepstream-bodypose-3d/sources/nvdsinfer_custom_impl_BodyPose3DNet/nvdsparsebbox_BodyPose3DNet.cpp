/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cmath>
#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include "nvdssample_BodyPose3DNet_common.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

/* This is a sample bounding box parsing function for the sample BodyPose3DNet
 * detector model provided with the TensorRT samples. */

extern "C"
bool NvDsInferParseCustomBodyPose3DNet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomBodyPose3DNet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  static int bboxPredLayerIndex = -1;
  static int clsProbLayerIndex = -1;
  static int roisLayerIndex = -1;
  static const int NUM_CLASSES_FASTER_RCNN = 21;
  static bool classMismatchWarn = false;
  int numClassesToParse;

  if (bboxPredLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "bbox_pred") == 0) {
        bboxPredLayerIndex = i;
        break;
      }
    }
    if (bboxPredLayerIndex == -1) {
    std::cerr << "Could not find bbox_pred layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (clsProbLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "cls_prob") == 0) {
        clsProbLayerIndex = i;
        break;
      }
    }
    if (clsProbLayerIndex == -1) {
    std::cerr << "Could not find cls_prob layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (roisLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "rois") == 0) {
        roisLayerIndex = i;
        break;
      }
    }
    if (roisLayerIndex == -1) {
    std::cerr << "Could not find rois layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (!classMismatchWarn) {
    if (NUM_CLASSES_FASTER_RCNN !=
        detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        NUM_CLASSES_FASTER_RCNN << std::endl;
    }
    classMismatchWarn = true;
  }

  numClassesToParse = MIN (NUM_CLASSES_FASTER_RCNN,
      detectionParams.numClassesConfigured);

  float *rois = (float *) outputLayersInfo[roisLayerIndex].buffer;
  float *deltas = (float *) outputLayersInfo[bboxPredLayerIndex].buffer;
  float *scores = (float *) outputLayersInfo[clsProbLayerIndex].buffer;

  for (int i = 0; i < nmsMaxOut; ++i)
  {
    float width = rois[i * 4 + 2] - rois[i * 4] + 1;
    float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
    float ctr_x = rois[i * 4] + 0.5f * width;
    float ctr_y = rois[i * 4 + 1] + 0.5f * height;
    float *deltas_offset = deltas + i * NUM_CLASSES_FASTER_RCNN * 4;
    for (int j = 0; j < numClassesToParse; ++j)
    {
      float confidence = scores[i * NUM_CLASSES_FASTER_RCNN + j];
      if (confidence < detectionParams.perClassPreclusterThreshold[j])
        continue;
      NvDsInferObjectDetectionInfo object;

      float dx = deltas_offset[j * 4];
      float dy = deltas_offset[j * 4 + 1];
      float dw = deltas_offset[j * 4 + 2];
      float dh = deltas_offset[j * 4 + 3];
      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = exp(dw) * width;
      float pred_h = exp(dh) * height;
      float rectx1 = MIN (pred_ctr_x - 0.5f * pred_w, networkInfo.width - 1.f);
      float recty1 = MIN (pred_ctr_y - 0.5f * pred_h, networkInfo.height - 1.f);
      float rectx2 = MIN (pred_ctr_x + 0.5f * pred_w, networkInfo.width - 1.f);
      float recty2 = MIN (pred_ctr_y + 0.5f * pred_h, networkInfo.height - 1.f);


      object.classId = j;
      object.detectionConfidence = confidence;

      /* Clip object box co-ordinates to network resolution */
      object.left = CLIP(rectx1, 0, networkInfo.width - 1);
      object.top = CLIP(recty1, 0, networkInfo.height - 1);
      object.width = CLIP(rectx2, 0, networkInfo.width - 1) - object.left + 1;
      object.height = CLIP(recty2, 0, networkInfo.height - 1) - object.top + 1;

      objectList.push_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBodyPose3DNet);

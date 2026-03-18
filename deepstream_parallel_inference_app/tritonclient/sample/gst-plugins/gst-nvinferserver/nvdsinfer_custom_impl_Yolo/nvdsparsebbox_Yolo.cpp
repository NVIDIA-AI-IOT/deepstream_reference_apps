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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"

static const int NUM_CLASSES_YOLO = 80;

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    const NvDsInferLayerInfo *boxes = nullptr;
    const NvDsInferLayerInfo *scores = nullptr;
    const NvDsInferLayerInfo *num = nullptr;
    const NvDsInferLayerInfo *classes_layer = nullptr;

    for (size_t l = 0; l < outputLayersInfo.size(); l++) {
        if (!strcmp(outputLayersInfo[l].layerName, "num_detections")) {
            num = &outputLayersInfo[l];
        }
        if (!strcmp(outputLayersInfo[l].layerName, "nmsed_boxes")) {
            boxes = &outputLayersInfo[l];
        }
        if (!strcmp(outputLayersInfo[l].layerName, "nmsed_scores")) {
            scores = &outputLayersInfo[l];
        }
        if (!strcmp(outputLayersInfo[l].layerName, "nmsed_classes")) {
            classes_layer = &outputLayersInfo[l];
        }
    }

    if (!boxes || !scores || !classes_layer) {
        std::cerr << "ERROR: Missing required output layers (nmsed_boxes, nmsed_scores, nmsed_classes)" << std::endl;
        return false;
    }

    const float* bbox_buffer = (const float*)boxes->buffer;
    const float* score_buffer = (const float*)scores->buffer;
    const float* class_buffer = (const float*)classes_layer->buffer;

    // Get number of detections
    uint num_bboxes;
    if (num) {
        num_bboxes = ((int*)num->buffer)[0];
    } else {
        num_bboxes = boxes->inferDims.d[0];
    }

    for (uint n = 0; n < num_bboxes; ++n) {
        int class_id = static_cast<int>(class_buffer[n]);

        // Validate class_id
        if (class_id < 0 || class_id >= (int)detectionParams.numClassesConfigured) {
            continue;
        }

        float score = score_buffer[n];
        if (score < detectionParams.perClassPreclusterThreshold[class_id]) {
            continue;
        }

        // Parse bbox: [x1, y1, x2, y2] normalized coordinates
        float bx1 = bbox_buffer[n * 4];
        float by1 = bbox_buffer[n * 4 + 1];
        float bx2 = bbox_buffer[n * 4 + 2];
        float by2 = bbox_buffer[n * 4 + 3];

        // Convert to pixel coordinates
        float x1 = clamp(bx1 * networkInfo.width, 0.0f, (float)networkInfo.width);
        float y1 = clamp(by1 * networkInfo.height, 0.0f, (float)networkInfo.height);
        float x2 = clamp(bx2 * networkInfo.width, 0.0f, (float)networkInfo.width);
        float y2 = clamp(by2 * networkInfo.height, 0.0f, (float)networkInfo.height);

        NvDsInferParseObjectInfo outObj;
        outObj.left = x1;
        outObj.top = y1;
        outObj.width = clamp(x2 - x1, 0.0f, (float)networkInfo.width);
        outObj.height = clamp(y2 - y1, 0.0f, (float)networkInfo.height);
        outObj.classId = class_id;
        outObj.rotation_angle = 0.0f;
        outObj.detectionConfidence = score;

        if (outObj.width < 1 || outObj.height < 1) {
            continue;
        }

        objectList.push_back(outObj);
    }

    return true;
}
/* YOLOv4 implementations end*/


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);

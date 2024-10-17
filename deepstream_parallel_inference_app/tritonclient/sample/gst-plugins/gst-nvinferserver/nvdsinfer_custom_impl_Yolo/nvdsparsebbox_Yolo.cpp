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

    std::vector<NvDsInferParseObjectInfo> objects;
    uint num_bboxes;
    float *score_buffer;
    float *bbox_buffer;
    
    
    for (int i=0; i<outputLayersInfo.size(); i++) {
        if (!strcmp(outputLayersInfo[i].layerName, "nmsed_boxes")) {
            const NvDsInferLayerInfo &boxes = outputLayersInfo[i];
            num_bboxes = boxes.inferDims.d[0];
            bbox_buffer = (float *)boxes.buffer;
        }
        else if (!strcmp(outputLayersInfo[i].layerName, "nmsed_scores")) {
            const NvDsInferLayerInfo &scores = outputLayersInfo[i];
            score_buffer = (float *)scores.buffer;
        }
    }

    for (int n=0; n<num_bboxes; n++) {
        NvDsInferParseObjectInfo outObj;
        if(score_buffer[n] > 0.1) {
            outObj.left = bbox_buffer[n*4] * networkInfo.width;
            outObj.top = bbox_buffer[n*4+1] * networkInfo.height;
            outObj.width = (bbox_buffer[n*4+2]-bbox_buffer[n*4]) * networkInfo.width;
            outObj.height = (bbox_buffer[n*4+3]-bbox_buffer[n*4+1]) * networkInfo.height;
            outObj.classId=n;
            outObj.detectionConfidence=score_buffer[n];
            objectList.push_back(outObj);
        }       
    }

    return true;
}
/* YOLOv4 implementations end*/


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);

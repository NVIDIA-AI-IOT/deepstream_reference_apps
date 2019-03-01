/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

unsigned clamp(const uint val, const uint minVal, const uint maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.left = x - bw / 2;
    b.width = bw;

    b.top = y - bh / 2;
    b.height = bh;

    b.left = clamp(b.left, 0, netW);
    b.width = clamp(b.width, 0, netW);
    b.top = clamp(b.top, 0, netH);
    b.height = clamp(b.height, 0, netH);

    return b;
}

void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (((bbi.left + bbi.width) > netW) || ((bbi.top + bbi.height) > netH)) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });
    std::vector<NvDsInferParseObjectInfo> out;
    for (auto i : binfo)
    {
        bool keep = true;
        for (auto j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

std::vector<NvDsInferParseObjectInfo> nmsAllClasses(const float nmsThresh,
                                                    std::vector<NvDsInferParseObjectInfo>& binfo,
                                                    const uint numClasses)
{
    std::vector<NvDsInferParseObjectInfo> result;
    std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.classId).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}

std::vector<NvDsInferParseObjectInfo>
decodeTensor(const float* detections, const std::vector<int> mask, const std::vector<float> anchors,
             const uint gridSize, const uint stride, const uint numBBoxes,
             const uint numOutputClasses, const float probThresh, const uint& netW,
             const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];

                const int numGridCells = gridSize * gridSize;
                const int bbindex = y * gridSize + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                if (maxProb > probThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
                }
            }
        }
    }
    return binfo;
}

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV3(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                           NvDsInferNetworkInfo const& networkInfo,
                                           NvDsInferParseDetectionParams const& detectionParams,
                                           std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static int outputBlobIndex1 = -1;
    static int outputBlobIndex2 = -1;
    static int outputBlobIndex3 = -1;
    static const int NUM_CLASSES_YOLO_V3 = 80;
    static bool classMismatchWarn = false;

    if (outputBlobIndex1 == -1)
    {
        for (uint i = 0; i < outputLayersInfo.size(); i++)
        {
            if (strcmp(outputLayersInfo[i].layerName, "yolo_83") == 0)
            {
                outputBlobIndex1 = i;
                break;
            }
        }
        if (outputBlobIndex1 == -1)
        {
            std::cerr << "Could not find output layer 'yolo_83' while parsing" << std::endl;
            return false;
        }
    }

    if (outputBlobIndex2 == -1)
    {
        for (uint i = 0; i < outputLayersInfo.size(); i++)
        {
            if (strcmp(outputLayersInfo[i].layerName, "yolo_95") == 0)
            {
                outputBlobIndex2 = i;
                break;
            }
        }
        if (outputBlobIndex2 == -1)
        {
            std::cerr << "Could not find output layer 'yolo_95' layer buffer while parsing"
                      << std::endl;
            return false;
        }
    }

    if (outputBlobIndex3 == -1)
    {
        for (uint i = 0; i < outputLayersInfo.size(); i++)
        {
            if (strcmp(outputLayersInfo[i].layerName, "yolo_107") == 0)
            {
                outputBlobIndex3 = i;
                break;
            }
        }
        if (outputBlobIndex3 == -1)
        {
            std::cerr << "Could not find output layer yolo_107 layer buffer while parsing"
                      << std::endl;
            return false;
        }
    }

    if (!classMismatchWarn)
    {
        if (NUM_CLASSES_YOLO_V3 != detectionParams.numClassesConfigured)
        {
            std::cerr << "WARNING: Num classes mismatch. Configured:"
                      << detectionParams.numClassesConfigured
                      << ", detected by network: " << NUM_CLASSES_YOLO_V3 << std::endl;
        }
        classMismatchWarn = true;
    }

    std::vector<float*> outputBlobs(3, nullptr);
    outputBlobs.at(0) = (float*) outputLayersInfo[outputBlobIndex1].buffer;
    outputBlobs.at(1) = (float*) outputLayersInfo[outputBlobIndex2].buffer;
    outputBlobs.at(2) = (float*) outputLayersInfo[outputBlobIndex3].buffer;

    const std::vector<float> kANCHORS
        = {10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
           45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    const std::vector<int> kMASK_1 = {6, 7, 8};
    const std::vector<int> kMASK_2 = {3, 4, 5};
    const std::vector<int> kMASK_3 = {0, 1, 2};
    const float kNMS_THRESH = 0.5f;
    const float kPROB_THRESH = 0.7f;
    const uint kNUM_BBOXES = 3;
    const uint kINPUT_H = 416;
    const uint kINPUT_W = 416;
    const uint kSTRIDE_1 = 32;
    const uint kSTRIDE_2 = 16;
    const uint kSTRIDE_3 = 8;
    const uint kGRID_SIZE_1 = kINPUT_H / kSTRIDE_1;
    const uint kGRID_SIZE_2 = kINPUT_H / kSTRIDE_2;
    const uint kGRID_SIZE_3 = kINPUT_H / kSTRIDE_3;

    std::vector<NvDsInferParseObjectInfo> objects;
    std::vector<NvDsInferParseObjectInfo> objects1
        = decodeTensor(outputBlobs.at(0), kMASK_1, kANCHORS, kGRID_SIZE_1, kSTRIDE_1, kNUM_BBOXES,
                       NUM_CLASSES_YOLO_V3, kPROB_THRESH, kINPUT_W, kINPUT_H);
    std::vector<NvDsInferParseObjectInfo> objects2
        = decodeTensor(outputBlobs.at(1), kMASK_2, kANCHORS, kGRID_SIZE_2, kSTRIDE_2, kNUM_BBOXES,
                       NUM_CLASSES_YOLO_V3, kPROB_THRESH, kINPUT_W, kINPUT_H);
    std::vector<NvDsInferParseObjectInfo> objects3
        = decodeTensor(outputBlobs.at(2), kMASK_3, kANCHORS, kGRID_SIZE_3, kSTRIDE_3, kNUM_BBOXES,
                       NUM_CLASSES_YOLO_V3, kPROB_THRESH, kINPUT_W, kINPUT_H);
    objects.insert(objects.end(), objects1.begin(), objects1.end());
    objects.insert(objects.end(), objects2.begin(), objects2.end());
    objects.insert(objects.end(), objects3.begin(), objects3.end());
    objectList.clear();
    objectList = nmsAllClasses(kNMS_THRESH, objects, NUM_CLASSES_YOLO_V3);

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV3);

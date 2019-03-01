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
#include <iostream>

/* C-linkage to prevent name-mangling */
extern "C" bool
NvDsInferParseCustomCaffeMNIST(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                               NvDsInferNetworkInfo const& networkInfo,
                               NvDsInferParseDetectionParams const& detectionParams,
                               std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static int outputBlobIndex1 = -1;
    static const int NUM_CLASSES_CAFFE_MNIST = 10;
    static bool classMismatchWarn = false;

    if (outputBlobIndex1 == -1)
    {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
        {
            if (strcmp(outputLayersInfo[i].layerName, "prob") == 0)
            {
                outputBlobIndex1 = i;
                break;
            }
        }
        if (outputBlobIndex1 == -1)
        {
            std::cerr << "Could not find output layer 'prob' while parsing" << std::endl;
            return false;
        }
    }

    if (!classMismatchWarn)
    {
        if (NUM_CLASSES_CAFFE_MNIST != detectionParams.numClassesConfigured)
        {
            std::cerr << "WARNING: Num classes mismatch. Configured:"
                      << detectionParams.numClassesConfigured
                      << ", detected by network: " << NUM_CLASSES_CAFFE_MNIST << std::endl;
        }
        classMismatchWarn = true;
    }

    const float* outputBlob = static_cast<const float*>(outputLayersInfo[outputBlobIndex1].buffer);
    const int maxIndex
        = std::max_element(outputBlob, outputBlob + NUM_CLASSES_CAFFE_MNIST) - outputBlob;

    NvDsInferParseObjectInfo object;
    object.classId = maxIndex;
    object.detectionConfidence = outputBlob[maxIndex];

    object.left = networkInfo.width / 8;
    object.top = networkInfo.height / 8;
    object.width = networkInfo.width * 6 / 8;
    ;
    object.height = networkInfo.height * 6 / 8;

    objectList.push_back(object);

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCaffeMNIST);

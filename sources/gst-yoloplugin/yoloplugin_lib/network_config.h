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

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sys/time.h>
#include <vector>

// Uncomment the macro to choose a type of network
#define MODEL_V2
// #define MODEL_V3

// Common global vars
extern const bool PRINT_PRED_INFO;
extern const bool PRINT_PERF_INFO;

extern const std::string PRECISION;
extern const std::string INPUT_BLOB_NAME;
extern const uint INPUT_H;
extern const uint INPUT_W;
extern const uint INPUT_C;
extern const uint64_t INPUT_SIZE;
extern const uint OUTPUT_CLASSES;
extern const float PROB_THRESH;
extern const float NMS_THRESH;
extern const std::vector<std::string> CLASS_NAMES;
extern const uint BBOXES;
extern const std::vector<float> ANCHORS;

extern const std::string MODELS_PATH;
extern const std::string YOLO_CONFIG_PATH;
extern const std::string TRAINED_WEIGHTS_PATH;
extern const std::string NETWORK_TYPE;
extern const std::string CALIB_TABLE_PATH;
extern const std::string CALIBRATION_SET;

// Model V2 specific global vars
extern const uint STRIDE;
extern const uint GRID_SIZE;
extern const uint64_t OUTPUT_SIZE;
extern const std::string OUTPUT_BLOB_NAME;

// Model V3 specific global vars
extern const uint STRIDE_1;
extern const uint STRIDE_2;
extern const uint STRIDE_3;
extern const uint GRID_SIZE_1;
extern const uint GRID_SIZE_2;
extern const uint GRID_SIZE_3;
extern const uint64_t OUTPUT_SIZE_1;
extern const uint64_t OUTPUT_SIZE_2;
extern const uint64_t OUTPUT_SIZE_3;
extern const std::vector<int> MASK_1;
extern const std::vector<int> MASK_2;
extern const std::vector<int> MASK_3;
extern const std::string OUTPUT_BLOB_NAME_1;
extern const std::string OUTPUT_BLOB_NAME_2;
extern const std::string OUTPUT_BLOB_NAME_3;

#endif //_NETWORK_H_
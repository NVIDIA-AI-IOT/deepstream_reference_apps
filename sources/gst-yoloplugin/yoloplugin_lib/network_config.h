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

namespace config
{
// Common global vars
extern const bool kPRINT_PRED_INFO;
extern const bool kPRINT_PERF_INFO;
extern const bool kSAVE_DETECTIONS;

extern const std::string kPRECISION;
extern const std::string kINPUT_BLOB_NAME;
extern const uint kINPUT_H;
extern const uint kINPUT_W;
extern const uint kINPUT_C;
extern const uint64_t kINPUT_SIZE;
extern const uint kOUTPUT_CLASSES;
extern const float kPROB_THRESH;
extern const float kNMS_THRESH;
extern const std::vector<std::string> kCLASS_NAMES;
extern const uint kBBOXES;
extern const std::vector<float> kANCHORS;

extern const std::string kMODELS_PATH;
extern const std::string kDETECTION_RESULTS_PATH;
extern const std::string kYOLO_CONFIG_PATH;
extern const std::string kTRAINED_WEIGHTS_PATH;
extern const std::string kNETWORK_TYPE;
extern const std::string kCALIB_TABLE_PATH;
extern const std::string kCALIBRATION_SET;
extern const std::string kTEST_IMAGES;

// Model V2 specific global vars
extern const uint kSTRIDE;
extern const uint kGRID_SIZE;
extern const uint64_t kOUTPUT_SIZE;
extern const std::string kOUTPUT_BLOB_NAME;

// Model V3 specific global vars
extern const uint kSTRIDE_1;
extern const uint kSTRIDE_2;
extern const uint kSTRIDE_3;
extern const uint kGRID_SIZE_1;
extern const uint kGRID_SIZE_2;
extern const uint kGRID_SIZE_3;
extern const uint64_t kOUTPUT_SIZE_1;
extern const uint64_t kOUTPUT_SIZE_2;
extern const uint64_t kOUTPUT_SIZE_3;
extern const std::vector<int> kMASK_1;
extern const std::vector<int> kMASK_2;
extern const std::vector<int> kMASK_3;
extern const std::string kOUTPUT_BLOB_NAME_1;
extern const std::string kOUTPUT_BLOB_NAME_2;
extern const std::string kOUTPUT_BLOB_NAME_3;

} // namespace config

#endif //_NETWORK_H_
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

namespace config
{
// Common global vars
extern const unsigned int kBATCHSIZE;
extern const std::string kPRECISION;
extern const std::string kINPUT_BLOB_NAME;
extern const std::string kOUTPUT_BLOB_NAME;
extern const uint kINPUT_H;
extern const uint kINPUT_W;
extern const uint kINPUT_C;
extern const uint kOUTPUTSIZE;
extern const uint kMAXWORKSPACESIZE;

extern const std::string kTRAINED_WEIGHTS_PATH;
extern const std::string kPLAN_FILE_PATH;
extern const std::string kCALIB_TABLE_PATH;
extern const std::string kIMAGE_DATASET_DIR;
} // namespace config

#endif //_NETWORK_H_
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

#include "network_config.h"

namespace config
{
// Common global vars
    const unsigned int kBATCHSIZE = 1;
    const std::string kPRECISION = "kFLOAT"; //kINT8 for lower precision
    const std::string kINPUT_BLOB_NAME = "data";
    const std::string kOUTPUT_BLOB_NAME = "linear";
    const uint kINPUT_H = 224;
    const uint kINPUT_W = 224;
    const uint kINPUT_C = 3;
    const uint kOUTPUTSIZE = 1000;
    const uint kMAXWORKSPACESIZE = 1 << 30;

    const std::string kTRAINED_WEIGHTS_PATH = "data/SE-ResNet50.wts";
    const std::string kPLAN_FILE_PATH = "data/SE-ResNet50-" + kPRECISION + "-batch" + std::to_string(kBATCHSIZE) + ".engine";
    const std::string kCALIB_TABLE_PATH = "lib/calibration/se-resnet50-calibration.table";
    const std::string kIMAGE_DATASET_DIR = "/path/to/ImageNet/dataset/";
} // namespace config

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

#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

/* OpenCV headers */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <experimental/filesystem>

#include "NvInfer.h"
#include "common.h"
#include "ds_image.h"
static Logger gLogger;
class DsImage;

// Common helper functions
cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages,
                         const int& inputH,
                         const int& inputW);

bool fileExists(const std::string fileName);

std::string locateFile(const std::string& input);

std::vector<std::pair<std::string, uint32_t>> loadImageList( std::string fileName);

std::vector<std::string> getSynsets( std::string fileName);

nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, nvinfer1::ILogger& logger);

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

int getNumChannels(nvinfer1::ITensor* t);
int getHight(nvinfer1::ITensor* t);
int getWidth(nvinfer1::ITensor* t);
void progBar(int index, int maxinum);

// Helper functions to create ResNet-SE engine
nvinfer1::IScaleLayer* addBN( nvinfer1::INetworkDefinition* network, 
                              nvinfer1::ITensor& input, std::map<std::string, 
                              nvinfer1::Weights>& weightMap, const std::string name);

#endif

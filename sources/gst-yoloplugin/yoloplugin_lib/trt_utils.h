
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
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NvInfer.h"

#include "ds_image.h"
#include "plugin_factory.h"

class DsImage;
struct BBox
{
    float x1, y1, x2, y2;
};

struct BBoxInfo
{
    BBox box;
    int label;
    float prob;
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

inline float sigmoid(const float& x) { return 1.0f / (1.0f + exp(-x)); }

// Common helper functions
cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages, const int& inputH,
                         const int& inputW);
std::string trim(std::string s);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName);
BBox convertBBox(const float& bx, const float& by, const float& bw, const float& bh,
                 const int& stride);
void printPredictions(const BBoxInfo& info, const std::string& className);
std::vector<std::string> loadImageList(const std::string filename);
std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);
nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, PluginFactory* pluginFactory);
std::vector<std::map<std::string, std::string>> parseConfig(const std::string cfgFilePath);
void displayConfig(const std::vector<std::map<std::string, std::string>>& blocks);
std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType);
std::string dimsToString(const nvinfer1::Dims d);
void displayDimType(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);

// Helper functions to create yolo engine
nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                    int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr);

#endif

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

#ifndef _YOLO_H_
#define _YOLO_H_

#include "calibrator.h"
#include "plugin_factory.h"
#include "trt_utils.h"

#include "NvInfer.h"

#include <stdint.h>
#include <string>
#include <vector>

/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo
{
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string labelsFilePath;
    std::string precision;
    std::string calibrationTablePath;
    std::string enginePath;
    std::string inputBlobName;
};

/**
 * Holds information about runtime inference params.
 */
struct InferParams
{
    bool printPerfInfo;
    bool printPredictionInfo;
    std::string calibrationImages;
    float probThresh;
    float nmsThresh;
};

/**
 * Holds information about an output tensor of the yolo network.
 */
struct TensorInfo
{
    std::string blobName;
    uint stride{0};
    uint gridSize{0};
    uint numClasses{0};
    uint numBBoxes{0};
    uint64_t volume{0};
    std::vector<uint> masks;
    std::vector<float> anchors;
    int bindingIndex{-1};
    float* hostBuffer{nullptr};
};

class Yolo
{
public:
    std::string getNetworkType() const { return m_NetworkType; }
    float getNMSThresh() const { return m_NMSThresh; }
    std::string getClassName(const int& label) const { return m_ClassNames.at(label); }
    int getInputH() const { return m_InputH; }
    int getInputW() const { return m_InputW; }
    bool isPrintPredictions() const { return m_PrintPredictions; }
    bool isPrintPerfInfo() const { return m_PrintPerfInfo; }
    void doInference(const unsigned char* input);
    std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH,
                                           const int& imageW);
    virtual std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH,
                                               const int imageW, const TensorInfo& tensor)
        = 0;
    virtual ~Yolo();

protected:
    Yolo(const uint batchSize, const NetworkInfo& networkInfo, const InferParams& inferParams);
    std::string m_EnginePath;
    const std::string m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_LabelsFilePath;
    const std::string m_Precision;
    const std::string m_CalibImagesFilePath;
    std::string m_CalibTableFilePath;
    const std::string m_InputBlobName;
    std::vector<TensorInfo> m_OutputTensors;
    std::vector<std::map<std::string, std::string>> m_configBlocks;
    uint m_InputH;
    uint m_InputW;
    uint m_InputC;
    uint64_t m_InputSize;
    const float m_ProbThresh;
    const float m_NMSThresh;
    std::vector<std::string> m_ClassNames;
    const bool m_PrintPerfInfo;
    const bool m_PrintPredictions;
    Logger m_Logger;

    // TRT specific members
    const uint m_BatchSize;
    nvinfer1::ICudaEngine* m_Engine;
    nvinfer1::IExecutionContext* m_Context;
    std::vector<void*> m_DeviceBuffers;
    int m_InputBindingIndex;
    cudaStream_t m_CudaStream;
    PluginFactory* m_PluginFactory;
    std::unique_ptr<YoloTinyMaxpoolPaddingFormula> m_TinyMaxpoolPaddingFormula;

private:
    void createYOLOEngine(const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
                          Int8EntropyCalibrator* calibrator = nullptr);
    std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath);
    void parseConfigBlocks();
    void allocateBuffers();
    bool verifyYoloEngine();
};

#endif // _YOLO_H_
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

class Yolo
{
public:
    std::string getNetworkType() const { return m_NetworkType; }
    float getNMSThresh() const { return m_NMSThresh; }
    std::string getClassName(const int& label) const { return m_ClassNames.at(label); }
    std::string getCalibTableFilePath() const { return m_CalibTableFilePath; }
    int getInputH() const { return m_InputH; }
    int getInputW() const { return m_InputW; }
    bool isPrintPredictions() const { return m_PrintPredictions; }
    bool isPrintPerfInfo() const { return m_PrintPerfInfo; }
    virtual void doInference(const unsigned char* input) = 0;
    virtual std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH,
                                                   const int& imageW)
        = 0;
    virtual ~Yolo();

protected:
    explicit Yolo(const uint batchSize);
    const std::string m_ModelsPath;
    const std::string m_ConfigFilePath;
    const std::string m_TrainedWeightsPath;
    const std::string m_NetworkType;
    const std::string m_CalibImagesFilePath;
    const std::string m_CalibTableFilePath;
    const std::string m_Precision;
    const std::string m_InputBlobName;
    const uint m_InputH;
    const uint m_InputW;
    const uint m_InputC;
    const uint64_t m_InputSize;
    const uint m_NumOutputClasses;
    const uint m_NumBBoxes;
    const float m_ProbThresh;
    const float m_NMSThresh;
    const std::vector<float> m_Anchors;
    const std::vector<std::string> m_ClassNames;
    const bool m_PrintPerfInfo;
    const bool m_PrintPredictions;
    Logger m_Logger;

    // TRT specific members
    const uint m_BatchSize;
    nvinfer1::ICudaEngine* m_Engine;
    nvinfer1::IExecutionContext* m_Context;
    std::vector<void*> m_Bindings;
    std::vector<float*> m_TrtOutputBuffers;
    int m_InputIndex;
    cudaStream_t m_CudaStream;
    PluginFactory* m_PluginFactory;

private:
    void createYOLOEngine(const int batchSize, const std::string yoloConfigPath,
                          const std::string trainedWeightsPath, const std::string planFilePath,
                          const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
                          Int8EntropyCalibrator* calibrator = nullptr);
};

#endif // _YOLO_H_
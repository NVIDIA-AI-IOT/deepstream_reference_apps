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

#ifndef _SE_RESNET50_H_
#define _SE_RESNET50_H_

#include "calibrator.h"
#include "trt_utils.h"
#include "network_config.h"

class SE_ResNet50
{
    public:
        void doInference( const unsigned char* input, 
                          const int batchSize, 
                          float* output);
        int getInputH() const { return m_InputH; }
        int getInputW() const { return m_InputW; }
        int getInputC() const { return m_InputC; }
        unsigned int getOutputSize() const { return m_OutputSize; }
        std::string getCalibTableFilePath() const { return m_CalibTableFilePath; }
        virtual ~SE_ResNet50();    
        explicit SE_ResNet50();

    private:
        const std::string m_Precision;
        const unsigned int m_BatchSize;
        const int m_InputC;
        const int m_InputH;
        const int m_InputW;
        const unsigned int m_OutputSize;
        int inputIndex;
        int outputIndex;
        const std::string m_InputBlobName;
        const std::string m_OutputBlobName;
        const std::string m_PlanFilePath;
        const std::string m_CalibImagesFileDir;
        const std::string m_CalibTableFilePath;
        const std::string m_WeightFilePath;
        const int m_MaxWorkSpaceSize;
        nvinfer1::IExecutionContext* m_context;
        cudaStream_t m_stream;
        void* buffers[2];
        void createSE_Resnet50Engine(const unsigned int maxBatchSize, 
                                  const DataType dataType, 
                                  const std::string trainedWeightsPath, 
                                  nvinfer1::IInt8EntropyCalibrator* calibrator = nullptr);
};

#endif
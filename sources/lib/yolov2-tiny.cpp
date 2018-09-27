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

#include "yolov2-tiny.h"
#include "network_config.h"

YoloV2Tiny::YoloV2Tiny(uint batchSize) :
    Yolo(batchSize),
    m_Stride(config::yoloV2Tiny::kSTRIDE),
    m_GridSize(config::yoloV2Tiny::kGRID_SIZE),
    m_OutputIndex(-1),
    m_OutputSize(config::yoloV2Tiny::kOUTPUT_SIZE),
    m_OutputBlobName(config::yoloV2Tiny::kOUTPUT_BLOB_NAME)
{
    assert(m_NetworkType == "yolov2-tiny");
    // Allocate Buffers
    m_OutputIndex = m_Engine->getBindingIndex(m_OutputBlobName.c_str());
    assert(m_OutputIndex != -1);
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_InputIndex), m_BatchSize * m_InputSize * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex), m_BatchSize * m_OutputSize * sizeof(float)));
    m_TrtOutputBuffers.front() = new float[m_OutputSize * m_BatchSize];
};

void YoloV2Tiny::doInference(const unsigned char* input)
{
    NV_CUDA_CHECK(cudaMemcpyAsync(m_Bindings.at(m_InputIndex), input,
                                  m_BatchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice,
                                  m_CudaStream));

    m_Context->enqueue(m_BatchSize, m_Bindings.data(), m_CudaStream, nullptr);

    NV_CUDA_CHECK(cudaMemcpyAsync(m_TrtOutputBuffers.at(0), m_Bindings.at(m_OutputIndex),
                                  m_BatchSize * m_OutputSize * sizeof(float),
                                  cudaMemcpyDeviceToHost, m_CudaStream));
    cudaStreamSynchronize(m_CudaStream);
}

std::vector<BBoxInfo> YoloV2Tiny::decodeDetections(const int& imageIdx, const int& imageH,
                                                   const int& imageW)
{
    std::vector<BBoxInfo> binfo;
    const float* detections = &m_TrtOutputBuffers.at(0)[imageIdx * m_OutputSize];
    float scalingFactor
        = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
    for (uint y = 0; y < m_GridSize; y++)
    {
        for (uint x = 0; x < m_GridSize; x++)
        {
            for (uint b = 0; b < m_NumBBoxes; b++)
            {
                const float pw = m_Anchors[2 * b];
                const float ph = m_Anchors[2 * b + 1];
                const int numGridCells = m_GridSize * m_GridSize;
                const int bbindex = y * m_GridSize + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 1)];
                const float bw = pw
                    * exp(detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 2)]);
                const float bh = ph
                    * exp(detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 3)]);

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 4)];
                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < m_NumOutputClasses; i++)
                {
                    float prob
                        = detections[bbindex
                                     + numGridCells * (b * (5 + m_NumOutputClasses) + (5 + i))];

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }

                maxProb = objectness * maxProb;

                if (maxProb > m_ProbThresh)
                {
                    BBoxInfo bbi;
                    bbi.box = convertBBox(bx, by, bw, bh, m_Stride);

                    // Undo Letterbox
                    float xCorrection = (m_InputW - scalingFactor * imageW) / 2;
                    float yCorrection = (m_InputH - scalingFactor * imageH) / 2;
                    bbi.box.x1 -= xCorrection;
                    bbi.box.x2 -= xCorrection;
                    bbi.box.y1 -= yCorrection;
                    bbi.box.y2 -= yCorrection;

                    // Restore to input image resolution
                    bbi.box.x1 /= scalingFactor;
                    bbi.box.x2 /= scalingFactor;
                    bbi.box.y1 /= scalingFactor;
                    bbi.box.y2 /= scalingFactor;

                    bbi.box.x1 = clamp(bbi.box.x1, 0, imageW);
                    bbi.box.x2 = clamp(bbi.box.x2, 0, imageW);
                    bbi.box.y1 = clamp(bbi.box.y1, 0, imageH);
                    bbi.box.y2 = clamp(bbi.box.y2, 0, imageH);

                    bbi.label = maxIndex;
                    bbi.prob = maxProb;

                    binfo.push_back(bbi);
                }
            }
        }
    }
    return binfo;
}
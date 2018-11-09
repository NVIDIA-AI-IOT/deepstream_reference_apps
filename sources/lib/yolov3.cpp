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
#include "yolov3.h"
#include "network_config.h"

YoloV3::YoloV3(uint batchSize) :
    Yolo(batchSize),
    m_Stride1(config::yoloV3::kSTRIDE_1),
    m_Stride2(config::yoloV3::kSTRIDE_2),
    m_Stride3(config::yoloV3::kSTRIDE_3),
    m_GridSize1(config::yoloV3::kGRID_SIZE_1),
    m_GridSize2(config::yoloV3::kGRID_SIZE_2),
    m_GridSize3(config::yoloV3::kGRID_SIZE_3),
    m_OutputIndex1(-1),
    m_OutputIndex2(-1),
    m_OutputIndex3(-1),
    m_OutputSize1(config::yoloV3::kOUTPUT_SIZE_1),
    m_OutputSize2(config::yoloV3::kOUTPUT_SIZE_2),
    m_OutputSize3(config::yoloV3::kOUTPUT_SIZE_3),
    m_Mask1(config::yoloV3::kMASK_1),
    m_Mask2(config::yoloV3::kMASK_2),
    m_Mask3(config::yoloV3::kMASK_3),
    m_OutputBlobName1(config::yoloV3::kOUTPUT_BLOB_NAME_1),
    m_OutputBlobName2(config::yoloV3::kOUTPUT_BLOB_NAME_2),
    m_OutputBlobName3(config::yoloV3::kOUTPUT_BLOB_NAME_3)
{
    assert(m_NetworkType == "yolov3");
    // Allocate Buffers
    m_OutputIndex1 = m_Engine->getBindingIndex(m_OutputBlobName1.c_str());
    assert(m_OutputIndex1 != -1);
    m_OutputIndex2 = m_Engine->getBindingIndex(m_OutputBlobName2.c_str());
    assert(m_OutputIndex2 != -1);
    m_OutputIndex3 = m_Engine->getBindingIndex(m_OutputBlobName3.c_str());
    assert(m_OutputIndex3 != -1);
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_InputIndex), m_BatchSize * m_InputSize * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex1), m_BatchSize * m_OutputSize1 * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex2), m_BatchSize * m_OutputSize2 * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex3), m_BatchSize * m_OutputSize3 * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMallocHost(&m_TrtOutputBuffers[0], m_OutputSize1 * m_BatchSize * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMallocHost(&m_TrtOutputBuffers[1], m_OutputSize2 * m_BatchSize * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMallocHost(&m_TrtOutputBuffers[2], m_OutputSize3 * m_BatchSize * sizeof(float)));
};

void YoloV3::doInference(const unsigned char* input)
{
    NV_CUDA_CHECK(cudaMemcpyAsync(m_Bindings.at(m_InputIndex), input,
                                  m_BatchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice,
                                  m_CudaStream));

    m_Context->enqueue(m_BatchSize, m_Bindings.data(), m_CudaStream, nullptr);

    NV_CUDA_CHECK(cudaMemcpyAsync(m_TrtOutputBuffers.at(0), m_Bindings.at(m_OutputIndex1),
                                  m_BatchSize * m_OutputSize1 * sizeof(float),
                                  cudaMemcpyDeviceToHost, m_CudaStream));
    NV_CUDA_CHECK(cudaMemcpyAsync(m_TrtOutputBuffers.at(1), m_Bindings.at(m_OutputIndex2),
                                  m_BatchSize * m_OutputSize2 * sizeof(float),
                                  cudaMemcpyDeviceToHost, m_CudaStream));
    NV_CUDA_CHECK(cudaMemcpyAsync(m_TrtOutputBuffers.at(2), m_Bindings.at(m_OutputIndex3),
                                  m_BatchSize * m_OutputSize3 * sizeof(float),
                                  cudaMemcpyDeviceToHost, m_CudaStream));

    cudaStreamSynchronize(m_CudaStream);
}

std::vector<BBoxInfo> YoloV3::decodeDetections(const int& imageIdx, const int& imageH,
                                               const int& imageW)
{
    std::vector<BBoxInfo> binfo;

    std::vector<BBoxInfo> binfo1
        = decodeTensor(imageH, imageW, &m_TrtOutputBuffers.at(0)[imageIdx * m_OutputSize1], m_Mask1,
                       m_GridSize1, m_Stride1);
    std::vector<BBoxInfo> binfo2
        = decodeTensor(imageH, imageW, &m_TrtOutputBuffers.at(1)[imageIdx * m_OutputSize2], m_Mask2,
                       m_GridSize2, m_Stride2);
    std::vector<BBoxInfo> binfo3
        = decodeTensor(imageH, imageW, &m_TrtOutputBuffers.at(2)[imageIdx * m_OutputSize3], m_Mask3,
                       m_GridSize3, m_Stride3);

    binfo.insert(binfo.end(), binfo1.begin(), binfo1.end());
    binfo.insert(binfo.end(), binfo2.begin(), binfo2.end());
    binfo.insert(binfo.end(), binfo3.begin(), binfo3.end());

    return binfo;
}

std::vector<BBoxInfo> YoloV3::decodeTensor(const int& imageH, const int& imageW,
                                           const float* detections, const std::vector<int> mask,
                                           const uint gridSize, const uint stride)
{
    float scalingFactor
        = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
    std::vector<BBoxInfo> binfo;
    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
            for (uint b = 0; b < m_NumBBoxes; ++b)
            {
                const float pw = m_Anchors[mask[b] * 2];
                const float ph = m_Anchors[mask[b] * 2 + 1];

                const int numGridCells = gridSize * gridSize;
                const int bbindex = y * gridSize + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + m_NumOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < m_NumOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + m_NumOutputClasses) + (5 + i))]);

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
                    bbi.box = convertBBox(bx, by, bw, bh, stride);

                    // Undo Letterbox
                    float x_correction = (m_InputW - scalingFactor * imageW) / 2;
                    float y_correction = (m_InputH - scalingFactor * imageH) / 2;
                    bbi.box.x1 -= x_correction;
                    bbi.box.x2 -= x_correction;
                    bbi.box.y1 -= y_correction;
                    bbi.box.y2 -= y_correction;

                    // Restore to input resolution
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

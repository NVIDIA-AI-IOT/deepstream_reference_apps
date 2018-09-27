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

#ifndef _YOLO_V3_TINY_
#define _YOLO_V3_TINY_

#include "yolo.h"

class YoloV3Tiny : public Yolo
{
public:
    explicit YoloV3Tiny(const uint batchSize);
    void doInference(const unsigned char* input) override;
    std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH,
                                           const int& imageW) override;

private:
    std::vector<BBoxInfo> decodeTensor(const int& imageH, const int& imageW,
                                       const float* dectections, std::vector<int> mask,
                                       const uint gridSize, const uint stride);
    const uint m_Stride1;
    const uint m_Stride2;
    const uint m_GridSize1;
    const uint m_GridSize2;
    int m_OutputIndex1;
    int m_OutputIndex2;
    const uint64_t m_OutputSize1;
    const uint64_t m_OutputSize2;
    const std::vector<int> m_Mask1;
    const std::vector<int> m_Mask2;
    const std::string m_OutputBlobName1;
    const std::string m_OutputBlobName2;
};

#endif // _YOLO_V3_TINY_
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

#ifndef _YOLO_V2_TINY_
#define _YOLO_V2_TINY_

#include "yolo.h"

class YoloV2Tiny : public Yolo
{
public:
    explicit YoloV2Tiny(const uint batchSize);
    void doInference(const unsigned char* input) override;
    std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH,
                                           const int& imageW) override;

private:
    const uint m_Stride;
    const uint m_GridSize;
    int m_OutputIndex;
    const uint64_t m_OutputSize;
    const std::string m_OutputBlobName;
};

#endif // _YOLO_V2_TINY_
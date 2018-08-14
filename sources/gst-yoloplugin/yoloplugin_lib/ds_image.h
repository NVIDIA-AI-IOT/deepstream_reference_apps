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
#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "trt_utils.h"

class DsImage
{
private:
    int m_Height;
    int m_Width;
    int m_XOffset;
    int m_YOffset;
    float m_ScalingFactor;
    std::string m_ImagePath;

    // unaltered original Image
    cv::Mat m_OrigImage;
    // letterboxed Image given to the network as input
    cv::Mat m_LetterboxImage;

public:
    DsImage();
    DsImage(const std::string& path, const int& inputH, const int& inputW);
    cv::Mat getLetterBoxedImage() const;
};

#endif

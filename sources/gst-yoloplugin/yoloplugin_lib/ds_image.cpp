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
#include "ds_image.h"

DsImage::DsImage(const std::string& path, const int& inputH, const int& inputW) :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0)
{
    m_OrigImage = cv::imread(path, CV_LOAD_IMAGE_COLOR);

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
        assert(!"Unable to open DsImage");

    m_Height = m_OrigImage.rows;
    m_Width = m_OrigImage.cols;

    // resize the DsImage with scale
    float dim = std::max(m_Height, m_Width);
    int resizeH = ((m_Height / dim) * inputH);
    int resizeW = ((m_Width / dim) * inputW);
    m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);

    // Additional checks for images with non even dims
    if ((inputW - resizeW) % 2) resizeW--;
    if ((inputH - resizeH) % 2) resizeH--;
    assert((inputW - resizeW) % 2 == 0);
    assert((inputH - resizeH) % 2 == 0);

    m_XOffset = (inputW - resizeW) / 2;
    m_YOffset = (inputH - resizeH) / 2;

    assert(2 * m_XOffset + resizeW == inputW);
    assert(2 * m_YOffset + resizeH == inputH);

    // resizing
    cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_CUBIC);
    // letterboxing
    cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
                       m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    m_LetterboxImage.convertTo(m_LetterboxImage, CV_32FC3, 1 / 255.0);
    // converting to RGB and NCHW format
    m_LetterboxImage = cv::dnn::blobFromImage(m_LetterboxImage);
}

const float* DsImage::getImageData() const { return m_LetterboxImage.ptr<float>(0); }

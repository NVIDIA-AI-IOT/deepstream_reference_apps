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
#include <experimental/filesystem>

DsImage::DsImage() :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0),
    m_ImageName()
{
}

DsImage::DsImage(const std::string& path, const int& inputH, const int& inputW) :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0),
    m_ImageName()
{
    assert(fileExists(path));
    m_ImageName = std::experimental::filesystem::path(path).stem().string();
    m_OrigImage = cv::imread(path, cv::IMREAD_COLOR);

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
    {
        std::cout << "Unable to open image : " << path << std::endl;
        assert(0);
    }

    if (m_OrigImage.channels() != 3)
    {
        std::cout << "Non RGB images are not supported : " << path << std::endl;
        assert(0);
    }

    m_Height = m_OrigImage.rows;
    m_Width = m_OrigImage.cols;

    //*** Resize the shorter edge to 256 before center crop ***//
    m_ScalingFactor = 256.0 * 1.0/static_cast<float>(std::min(m_Height, m_Width));
    int resizeH = 0, resizeW = 0;
    if(m_Height < m_Width)
    {
        resizeH = 256, resizeW = static_cast<int>(m_ScalingFactor * m_Width + 0.5);
        assert(resizeW >= 256);
    }
    else
    {
        resizeH = static_cast<int>(m_ScalingFactor * m_Height + 0.5), resizeW = 256;
        assert(resizeH >= 256);
    }
    cv::resize(m_OrigImage, m_ProcessedImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_CUBIC);

    //*** Center crop 224x224 from the resized image ***//
    m_XOffset = static_cast<int>((resizeW - inputW) / 2);
    m_YOffset = static_cast<int>((resizeH - inputH) / 2);
    // Setup a rectangle to crop region of interest
    cv::Rect m_ROI( m_XOffset, m_YOffset, inputH, inputW);
    // Crop the resized image to the image contained by the rectangle
    m_ProcessedImage = m_ProcessedImage(m_ROI);

    // convert to FP32
    m_ProcessedImage.convertTo(m_ProcessedImage, CV_32FC3, 1);
    // subtract mean (BGR)
    cv::subtract(m_ProcessedImage, cv::Scalar( 103.53, 116.28, 123.675), m_ProcessedImage);
    // divide by standard deviation (BGR)
    cv::divide(m_ProcessedImage, cv::Scalar(57.375, 57.120003, 58.395), m_ProcessedImage); // BGR
}

void DsImage::showImage(std::string version) const
{
    assert(version=="O"||version=="P");
    cv::namedWindow(m_ImageName);
    cv::imshow(m_ImageName.c_str(), ((version=="O")? m_OrigImage:m_ProcessedImage));
    cv::waitKey(0);
}
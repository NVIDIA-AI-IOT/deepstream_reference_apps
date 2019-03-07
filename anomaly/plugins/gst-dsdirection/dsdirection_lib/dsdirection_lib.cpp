////////////////////////////////////////////////////////////////////////////////
// MIT License
// 
// Copyright (C) 2019 NVIDIA CORPORATION
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////
#include "dsdirection_lib.h"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>

using namespace cv;

inline bool IsFlowCorrect(float x, float y)
{
    return !cvIsNaN(x) && !cvIsNaN(y) && fabs(x) < 1e9 && fabs(y) < 1e9;
}

float DsDirectionFindMaxRad(DsOpticalFlowMap *map)
{
    float maxrad = 0.0;

    maxrad = 1;
    Mat cvflow(map->rows, map->cols, CV_32FC2, map->data);

    for (int y = 0; y < map->rows; ++y)
    {
        for (int x = 0; x < map->cols; ++x)
        {
            Point2f u(cvflow.at<Vec2f>(y, x)[0], cvflow.at<Vec2f>(y, x)[0]);

            if (!IsFlowCorrect(u.x, u.y))
                continue;

            maxrad = max(maxrad, u.x * u.x + u.y * u.y);
        }
    }

    return sqrt(maxrad);
}

void DsDirectionEstimation(DsOpticalFlowMap *map, DsMotionObject *object, float maxrad)
{
    if (maxrad <= 0) return;
    
    Mat cvflow(map->rows, map->cols, CV_32FC2, map->data);

    Point2f sum(0.0, 0.0);
    int num = 0;
    int ymax = object->top+object->height;
    int xmax = object->left+object->width;
    for (int y = object->top; y < ymax; ++y)
    {
        for (int x = object->left; x < xmax; ++x)
        {
            Point2f u(cvflow.at<Vec2f>(y, x)[0], cvflow.at<Vec2f>(y, x)[1]);

            if (!IsFlowCorrect(u.x, u.y))
                continue;

            sum.x += u.x/maxrad;
            sum.y += u.y/maxrad;
            num++;
        }
    }

    if (num > 0)
    {
        sum.x /= num;
        sum.y /= num;
        if (sum.x != 0 && sum.y != 0)
        {
            object->radius = sqrt(sum.x * sum.x + sum.y * sum.y);
            object->angle = atan2(-sum.y, -sum.x) / (float)CV_PI;
        }
    }
}

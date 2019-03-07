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


#include <stdio.h>
#include <stdlib.h>
#include <npp.h>
#include "dsinternalmemory.h"
#include "dsopticalflow_lib.h"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include "pthread.h"

static pthread_mutex_t *p_mutex = NULL;

void critical_enter()
{
    if (p_mutex == NULL)
    {
        p_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
        pthread_mutex_init(p_mutex, NULL);
    }

    pthread_mutex_lock(p_mutex);
}

void critical_leave()
{
    pthread_mutex_unlock(p_mutex);
}

using namespace cv;

#define CHECK_NPP_STATUS_IN_LOOP(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    printf ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    break; \
  } \
} while (0)

#define CHECK_CUDA_STATUS_IN_LOOP(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    printf ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    break; \
  } \
} while (0)

#define RGB_BYTES_PER_PIXEL 3
#define Y_BYTES_PER_PIXEL   1
#define USE_GPU_MALLOC

#if defined(USE_GPU_MALLOC)
typedef cv::cuda::GpuMat ImageMat; 
#else
typedef cv::Mat ImageMat;
#endif

inline bool IsFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b ComputeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void DrawOpticalFlow(const Mat_<float> &flowx, const Mat_<float> &flowy, Mat &dst, float maxmotion = -1)
{
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion * maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!IsFlowCorrect(u))
                    continue;
 
                maxrad = max(maxrad, u.x * u.x + u.y * u.y);
            }
        }
    }
    maxrad = sqrt(maxrad);
    
    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (IsFlowCorrect(u))
                dst.at<Vec3b>(y, x) = ComputeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

struct DsOpticalFlowCtx
{
    DsOpticalFlowInitParams initParams;
    DsInternalMemoryPool *outpool;
    DsInternalMemoryPool *inpool;
    ImageMat prev_frame;
    cuda::GpuMat flow;
    int cvtype;
    Ptr<cuda::FarnebackOpticalFlow> algorithm;
    uint32_t frames;
};

DsOpticalFlowCtx* DsOpticalFlowCreate()
{
    DsOpticalFlowCtx* ctx = new DsOpticalFlowCtx;

    return ctx;
}

bool DsOpticalFlowCtxInit(DsOpticalFlowCtx* ctx, DsOpticalFlowInitParams *initParams)
{
    if (initParams == NULL) return false;

    ctx->initParams = *initParams;

    int rows = ctx->initParams.processing_height;
    int cols = ctx->initParams.processing_width;
    ctx->cvtype = CV_32FC2;
    ctx->flow = cuda::GpuMat(rows, cols, ctx->cvtype);
    ctx->algorithm = cuda::FarnebackOpticalFlow::create();
    ctx->frames = 0;
    ctx->outpool = DsInternalMemoryPoolInit(rows * cols * CV_ELEM_SIZE(ctx->cvtype),
                                            ctx->initParams.pool_size);
    ctx->inpool = DsInternalMemoryPoolInit(rows * cols * CV_ELEM_SIZE(CV_8UC1), 2,
#if defined(USE_GPU_MALLOC)
                                           true);
#else
                                           false);
#endif

    return true;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 */
static ImageMat GetGpuMat(DsOpticalFlowCtx *ctx, void *input_buf,
                               int input_width, int input_height)
{
    ImageMat cvmat;
    void* buffer = NULL;
    int output_width = ctx->initParams.processing_width;
    int output_height = ctx->initParams.processing_height;

    /* parameter check */
    if (ctx == NULL || (input_width <= 0) || (input_width <= 0)) 
    {
        return cvmat;
    }

    // source ROI
    NppiRect oSrcROI = 
    {  
        0, 0, input_width, input_height
    };
    NppiSize oSrcCropSize = { input_width, input_height };
    // Destination ROI
    NppiRect DstROI = 
    { 
        0, 0, output_width, output_height
    };
    // Calculate scaling ratio while maintaining aspect ratio
    double ratio = MIN(1.0*output_width/input_width, 1.0*output_height/input_height);

    do
    {
        buffer = DsInternalMemoryPoolRequest(ctx->inpool);
        if (buffer == NULL)
        {
            printf("Failed to request host memory\n");
            break;
        }

        // Memset the memory
        CHECK_CUDA_STATUS_IN_LOOP(cudaMemset(buffer, 0,
                                          output_width * output_height * Y_BYTES_PER_PIXEL),
                          "Failed to memset cuda buffer");

        const Npp8u *ptr_top_left_pixel_Y = (Npp8u *)input_buf;

        /* Scale Y plane for optical flow processing */
        CHECK_NPP_STATUS_IN_LOOP(nppiResizeSqrPixel_8u_C1R(ptr_top_left_pixel_Y,
                                                   oSrcCropSize,
                                                   input_width * Y_BYTES_PER_PIXEL,
                                                   oSrcROI, 
                                                   (Npp8u *)buffer,
                                                   output_width * Y_BYTES_PER_PIXEL,
                                                   DstROI, ratio, ratio, 0, 0, NPPI_INTER_LINEAR),
                         "Failed to scale Gray frame");

        cvmat = ImageMat(output_height, output_width,
                        CV_8UC1, buffer, output_width * Y_BYTES_PER_PIXEL);
    } while (0);

    return cvmat;
}

// In case of an actual processing library, processing on data will be completed
// in this function and output will be returned
bool DsOpticalFlowProcess(DsOpticalFlowCtx *ctx, void* nvbuf, 
                          int width, int height, DsOpticalFlowOutput* out)
{
    if (out == NULL || nvbuf == NULL) return false;

    ImageMat input = GetGpuMat(ctx, nvbuf, width, height);

    /* We have previous frame, do optical flow calculation */
    if (!ctx->prev_frame.empty())
    {
        out->cols = ctx->initParams.processing_width;
        out->rows = ctx->initParams.processing_height;
        out->data = DsInternalMemoryPoolRequest(ctx->outpool);
        out->elemSize = CV_ELEM_SIZE(ctx->cvtype);
        
        cuda::GpuMat d_frame0(ctx->prev_frame);
        cuda::GpuMat d_frame1(input);

        /* opencv optical flow function seems not work correctly in 
           multithread scenario even with a specific cuda::Stream for
           each thread, so we have to serialize the process */
        critical_enter();
        ctx->algorithm->calc(d_frame0, d_frame1, ctx->flow);
        critical_leave();
        ctx->frames++;

        if (out->data == NULL)
        {
            printf("Failed to acquire optical flow buffer\n");
            return false;
        }

        Mat cvflow(out->rows, out->cols, ctx->cvtype, out->data);
        ctx->flow.download(cvflow);

        if (out->rgb)
        {
            Mat planes[2];
            split(cvflow, planes);
            Mat cvrgb(out->rows, out->cols, CV_8UC3, out->rgb);
            DrawOpticalFlow(planes[0], planes[1], cvrgb, 10);
        }
    }
    
    void *prev_buf = ctx->prev_frame.data;
    DsInternalMemoryPoolReturn(ctx->inpool, prev_buf);
    ctx->prev_frame = input;

    return true;
}

void DsOpticalFlowFreeData(DsOpticalFlowCtx* ctx, void* data)
{
    DsInternalMemoryPoolReturn(ctx->outpool, data);
}

void DsOpticalFlowCtxDeinit(DsOpticalFlowCtx *ctx)
{
    if (!ctx->prev_frame.empty())
    {
        void *prev_buf = ctx->prev_frame.data;
        DsInternalMemoryPoolReturn(ctx->inpool, prev_buf);
    }

    DsInternalMemoryPoolDeinit(ctx->inpool);
    DsInternalMemoryPoolDeinit(ctx->outpool);
    delete ctx;
}

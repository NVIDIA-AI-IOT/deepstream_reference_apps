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

#include "yoloplugin_lib.h"
#include "network_config.h"

#ifdef MODEL_V2
#include "yolov2.h"
#endif

#ifdef MODEL_V3
#include "yolov3.h"
#endif

#include <iomanip>
#include <sys/time.h>

static void decodeBatchDetections(const YoloPluginCtx* ctx, std::vector<YoloPluginOutput*>& outputs)
{
    for (int p = 0; p < ctx->batchSize; ++p)
    {
        YoloPluginOutput* out = new YoloPluginOutput;
        std::vector<BBoxInfo> binfo = ctx->inferenceNetwork->decodeDetections(
            p, ctx->initParams.processingHeight, ctx->initParams.processingWidth);

        std::vector<BBoxInfo> remaining
            = nonMaximumSuppression(ctx->inferenceNetwork->getNMSThresh(), binfo);
        out->numObjects = remaining.size();
        assert(out->numObjects <= MAX_OBJECTS_PER_FRAME);
        for (uint j = 0; j < remaining.size(); ++j)
        {
            BBoxInfo b = remaining.at(j);
            YoloPluginObject obj;
            obj.left = static_cast<int>(b.box.x1);
            obj.top = static_cast<int>(b.box.y1);
            obj.width = static_cast<int>(b.box.x2 - b.box.x1);
            obj.height = static_cast<int>(b.box.y2 - b.box.y1);
            strcpy(obj.label, ctx->inferenceNetwork->getClassName(b.label).c_str());
            out->object[j] = obj;

            if (ctx->inferenceNetwork->isPrintPredictions())
            {
                printPredictions(b, ctx->inferenceNetwork->getClassName(b.label));
            }
        }
        outputs.at(p) = out;
    }
}

static void dsPreProcessBatchInput(const std::vector<cv::Mat*>& cvmats, cv::Mat& batchBlob,
                                   const int& processingHeight, const int& processingWidth,
                                   const int& inputH, const int& inputW)
{

    std::vector<cv::Mat> batch_images(
        cvmats.size(), cv::Mat(cv::Size(processingWidth, processingHeight), CV_8UC3));
    for (uint i = 0; i < cvmats.size(); ++i)
    {
        cv::Mat imageResize, imageBorder, imageFloat, inputImage;
        inputImage = *cvmats.at(i);
        int maxBorder = std::max(inputImage.size().width, inputImage.size().height);

        assert((maxBorder - inputImage.size().height) % 2 == 0);
        assert((maxBorder - inputImage.size().width) % 2 == 0);

        int yOffset = (maxBorder - inputImage.size().height) / 2;
        int xOffset = (maxBorder - inputImage.size().width) / 2;

        // Letterbox and resize to maintain aspect ratio
        cv::copyMakeBorder(inputImage, imageBorder, yOffset, yOffset, xOffset, xOffset,
                           cv::BORDER_CONSTANT, cv::Scalar(127.5, 127.5, 127.5));
        cv::resize(imageBorder, imageResize, cv::Size(inputW, inputH), 0, 0, cv::INTER_CUBIC);
        imageResize.convertTo(imageFloat, CV_32FC3, 1 / 255.0);
        batch_images.at(i) = imageFloat;
    }

    batchBlob = cv::dnn::blobFromImages(batch_images, 1.0, cv::Size(inputW, inputH),
                                        cv::Scalar(0.0, 0.0, 0.0), false, false);
}

YoloPluginCtx* YoloPluginCtxInit(YoloPluginInitParams* initParams, size_t batchSize)
{
    YoloPluginCtx* ctx = new YoloPluginCtx;
    ctx->initParams = *initParams;
    ctx->batchSize = batchSize;
    assert(ctx->batchSize > 0);

#ifdef MODEL_V2
    ctx->inferenceNetwork = new YoloV2(batchSize);
#endif

#ifdef MODEL_V3
    ctx->inferenceNetwork = new YoloV3(batchSize);
#endif

    return ctx;
}

std::vector<YoloPluginOutput*> YoloPluginProcess(YoloPluginCtx* ctx, std::vector<cv::Mat*>& cvmats)
{
    std::vector<YoloPluginOutput*> outputs = std::vector<YoloPluginOutput*>(cvmats.size(), nullptr);
    cv::Mat preprocessedImages;
    struct timeval preStart, preEnd, inferStart, inferEnd, postStart, postEnd;
    double preElapsed = 0.0, inferElapsed = 0.0, postElapsed = 0.0;

    if (cvmats.size() > 0)
    {
        gettimeofday(&preStart, NULL);
        dsPreProcessBatchInput(cvmats, preprocessedImages, ctx->initParams.processingWidth,
                               ctx->initParams.processingHeight, ctx->inferenceNetwork->getInputH(),
                               ctx->inferenceNetwork->getInputW());
        gettimeofday(&preEnd, NULL);

        gettimeofday(&inferStart, NULL);
        ctx->inferenceNetwork->doInference(preprocessedImages.data);
        gettimeofday(&inferEnd, NULL);

        gettimeofday(&postStart, NULL);
        decodeBatchDetections(ctx, outputs);
        gettimeofday(&postEnd, NULL);
    }

    // Perf calc
    if (ctx->inferenceNetwork->isPrintPerfInfo())
    {
        preElapsed
            = ((preEnd.tv_sec - preStart.tv_sec) + (preEnd.tv_usec - preStart.tv_usec) / 1000000.0)
            * (1000 / ctx->batchSize);
        inferElapsed = ((inferEnd.tv_sec - inferStart.tv_sec)
                        + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
            * (1000 / ctx->batchSize);
        postElapsed = ((postEnd.tv_sec - postStart.tv_sec)
                       + (postEnd.tv_usec - postStart.tv_usec) / 1000000.0)
            * (1000 / ctx->batchSize);

        ctx->inferTime += inferElapsed;
        ctx->preTime += preElapsed;
        ctx->postTime += postElapsed;
        ++ctx->batchCount;
    }
    return outputs;
}

void YoloPluginCtxDeinit(YoloPluginCtx* ctx)
{
    if (ctx->inferenceNetwork->isPrintPerfInfo())
    {
        std::cout << "DS Example Perf Summary " << std::endl;
        std::cout << "Batch Size : " << ctx->batchSize << std::endl;
        std::cout << std::fixed << std::setprecision(4)
                  << "PreProcess : " << ctx->preTime / ctx->batchCount
                  << " ms Inference : " << ctx->inferTime / ctx->batchCount
                  << " ms PostProcess : " << ctx->postTime / ctx->batchCount << " ms Total : "
                  << (ctx->preTime + ctx->postTime + ctx->inferTime) / ctx->batchCount
                  << " ms per Image" << std::endl;
    }

    delete ctx->inferenceNetwork;
    delete ctx;
}

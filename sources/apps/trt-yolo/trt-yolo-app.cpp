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
#include "network_config.h"
#include "trt_utils.h"
#include "yolo.h"

#ifdef MODEL_V2
#include "yolov2.h"
#endif

#ifdef MODEL_V3
#include "yolov3.h"
#endif

#include <experimental/filesystem>
#include <gflags/gflags.h>
#include <string>
#include <sys/time.h>

DEFINE_bool(decode, true, "Decode the detections");
DEFINE_int32(batch_size, 1, "Batch size for the inference engine.");
DEFINE_uint64(seed, std::time(0), "Seed for the random number generator");

int main(int argc, char** argv)
{
    srand(unsigned(FLAGS_seed));
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    std::unique_ptr<Yolo> inferNet{nullptr};

#ifdef MODEL_V2
    inferNet = std::unique_ptr<Yolo>{new YoloV2(FLAGS_batch_size)};
#endif

#ifdef MODEL_V3
    inferNet = std::unique_ptr<Yolo>{new YoloV3(FLAGS_batch_size)};
#endif

    std::vector<std::string> imageList = loadImageList(config::kTEST_IMAGES);
    imageList.resize(static_cast<int>(imageList.size() / FLAGS_batch_size) * FLAGS_batch_size);
    std::random_shuffle(imageList.begin(), imageList.end(), [](int i) { return rand() % i; });
    std::cout << "Total number of images used for inference : " << imageList.size() << std::endl;

    std::vector<DsImage> dsImages(FLAGS_batch_size);
    const int barWidth = 70;
    double inferElapsed = 0;
    int batchCount = imageList.size() / FLAGS_batch_size;

    // Batched inference loop
    for (uint loopIdx = 0; loopIdx < imageList.size(); loopIdx += FLAGS_batch_size)
    {
        // Load a new batch
        for (uint imageIdx = loopIdx; imageIdx < (loopIdx + FLAGS_batch_size); ++imageIdx)
        {
            dsImages.at(imageIdx - loopIdx) = DsImage(imageList.at(imageIdx), inferNet->getInputH(),
                                                      inferNet->getInputW());
        }

        cv::Mat trtInput = blobFromDsImages(dsImages, inferNet->getInputH(), inferNet->getInputW());
        struct timeval inferStart, inferEnd;
        gettimeofday(&inferStart, NULL);
        inferNet->doInference(trtInput.data);
        gettimeofday(&inferEnd, NULL);
        inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec)
                         + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
            * 1000 / FLAGS_batch_size;

        if (FLAGS_decode)
        {
            for (int imageIdx = 0; imageIdx < FLAGS_batch_size; ++imageIdx)
            {
                auto curImage = dsImages.at(imageIdx);
                auto binfo = inferNet->decodeDetections(imageIdx, curImage.getImageHeight(),
                                                        curImage.getImageWidth());
                auto remaining = nonMaximumSuppression(inferNet->getNMSThresh(), binfo);
                for (auto b : remaining)
                {
                    if (inferNet->isPrintPredictions())
                    {
                        printPredictions(b, inferNet->getClassName(b.label));
                    }
                    curImage.addBBox(b, inferNet->getClassName(b.label));
                }

                if (config::kSAVE_DETECTIONS)
                {
                    curImage.saveImageJPEG(config::kDETECTION_RESULTS_PATH);
                }
            }
        }

        std::cout << "[";
        int progress = ((loopIdx + FLAGS_batch_size) * 100) / imageList.size();
        progress = progress > 100 ? 100 : progress;
        int pos = (barWidth * progress) / 100;
        for (int i = 0; i < pos; ++i)
        {
            std::cout << "=";
        }
        if (pos < barWidth) std::cout << ">";
        for (int i = pos; i < barWidth; ++i)
        {
            std::cout << " ";
        }
        std::cout << "] " << progress << " %\r";
        std::cout.flush();
    }
    std::cout << std::endl
              << "Network Type : " << inferNet->getNetworkType()
              << "Precision : " << config::kPRECISION << " Batch Size : " << FLAGS_batch_size
              << " Inference time per image : " << inferElapsed / batchCount << " ms" << std::endl;
    return 0;
}

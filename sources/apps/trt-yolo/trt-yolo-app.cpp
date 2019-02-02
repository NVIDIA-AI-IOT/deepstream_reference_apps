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
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov2.h"
#include "yolov3.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>

int main(int argc, char** argv)
{
    // Flag set in the command line overrides the value in the flagfile
    gflags::SetUsageMessage(
        "Usage : trt-yolo-app --flagfile=</path/to/config_file.txt> --<flag>=value ...");

    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();
    std::string testImages = getTestImages();
    std::string testImagesPath = getTestImagesPath();
    bool decode = getDecode();
    bool doBenchmark = getDoBenchmark();
    bool viewDetections = getViewDetections();
    bool saveDetections = getSaveDetections();
    std::string saveDetectionsPath = getSaveDetectionsPath();
    uint batchSize = getBatchSize();
    bool shuffleTestSet = getShuffleTestSet();

    srand(unsigned(seed));

    std::unique_ptr<Yolo> inferNet{nullptr};
    if ((networkType == "yolov2") || (networkType == "yolov2-tiny"))
    {
        inferNet = std::unique_ptr<Yolo>{new YoloV2(batchSize, yoloInfo, yoloInferParams)};
    }
    else if ((networkType == "yolov3") || (networkType == "yolov3-tiny"))
    {
        inferNet = std::unique_ptr<Yolo>{new YoloV3(batchSize, yoloInfo, yoloInferParams)};
    }
    else
    {
        assert(false && "Unrecognised network_type. Network Type has to be one among the following : yolov2, yolov2-tiny, yolov3 and yolov3-tiny");
    }

    if (testImages.empty())
    {
        std::cout << "Enter a valid file path for test_images config param" << std::endl;
        return -1;
    }

    std::vector<std::string> imageList = loadImageList(testImages, testImagesPath);
    std::cout << "Total number of images used for inference : " << imageList.size() << std::endl;

    if (shuffleTestSet)
    {
        std::random_shuffle(imageList.begin(), imageList.end(), [](int i) { return rand() % i; });
    }
    std::vector<DsImage> dsImages;
    const int barWidth = 70;
    double inferElapsed = 0;

    std::ofstream fout;
    bool written = false;
    if (doBenchmark)
    {
        size_t extIndex = testImages.find_last_of(".txt");
        fout.open(testImages.substr(0, extIndex - 3) + "_" + networkType + "_" + precision
                  + "_results.json");
        fout << "[";
    }
    // Batched inference loop
    for (uint loopIdx = 0; loopIdx < imageList.size(); loopIdx += batchSize)
    {
        // Load a new batch
        dsImages.clear();
        for (uint imageIdx = loopIdx;
             imageIdx < std::min((loopIdx + batchSize), static_cast<uint>(imageList.size()));
             ++imageIdx)
        {
            dsImages.emplace_back(imageList.at(imageIdx), inferNet->getInputH(),
                                  inferNet->getInputW());
        }

        cv::Mat trtInput = blobFromDsImages(dsImages, inferNet->getInputH(), inferNet->getInputW());
        struct timeval inferStart, inferEnd;
        gettimeofday(&inferStart, NULL);
        inferNet->doInference(trtInput.data, dsImages.size());
        gettimeofday(&inferEnd, NULL);
        inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec)
                         + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
            * 1000;

        if (decode)
        {
            for (uint imageIdx = 0; imageIdx < dsImages.size(); ++imageIdx)
            {
                auto curImage = dsImages.at(imageIdx);
                auto binfo = inferNet->decodeDetections(imageIdx, curImage.getImageHeight(),
                                                        curImage.getImageWidth());
                auto remaining
                    = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
                for (auto b : remaining)
                {
                    if (inferNet->isPrintPredictions())
                    {
                        printPredictions(b, inferNet->getClassName(b.label));
                    }
                    curImage.addBBox(b, inferNet->getClassName(b.label));
                }

                if (saveDetections)
                {
                    curImage.saveImageJPEG(saveDetectionsPath);
                }

                if (viewDetections)
                {
                    curImage.showImage();
                }

                if (doBenchmark)
                {
                    std::string jsonString = curImage.exportJson();
                    if (jsonString == "") continue;
                    if (written)
                        fout << "," << jsonString;
                    else
                        fout << jsonString;
                    written = true;
                }
            }
        }

        std::cout << "[";
        int progress = ((loopIdx + dsImages.size()) * 100) / imageList.size();
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
    if (doBenchmark)
    {
        fout << std::endl << "]";
        fout.close();
        std::cout << std::endl;
    }
    std::cout << std::endl
              << "Network Type : " << inferNet->getNetworkType() << " Precision : " << precision
              << " Batch Size : " << batchSize
              << " Inference time per image : " << inferElapsed / imageList.size() << " ms"
              << std::endl;

    return 0;
}

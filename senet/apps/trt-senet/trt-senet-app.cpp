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

#include "NvInfer.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <map>
#include <sstream>
#include <unistd.h>

#include "ds_image.h"
#include "trt_utils.h"
#include "se_resnet50.h"

using namespace nvinfer1;

int main(int argc, char** argv)
{
    static unsigned int BATCHSIZE = config::kBATCHSIZE;
    static std::string DATASETDIR = config::kIMAGE_DATASET_DIR;
    // Create inference engine
    std::unique_ptr<SE_ResNet50> inferNet = std::unique_ptr<SE_ResNet50>( new SE_ResNet50());

    // Read input image list
    std::cout<< "Reading input image list..." <<std::endl;
    std::vector<pair<std::string, uint32_t>> imgList = loadImageList(DATASETDIR + "val.txt");
    assert(imgList.size()!=0);

    std::cout<< "Reading synsets..." <<std::endl;
    std::vector<std::string> synsets = getSynsets(DATASETDIR + "synsets.txt");
    assert(synsets.size()!=0);

    // Do inference on all images in imgList
    uint32_t imgListSize = imgList.size();

    // Run Inference
    std::cout<< "Running inference..." <<std::endl;
    float prob[inferNet->getOutputSize() * BATCHSIZE];
    std::chrono::duration<double> diff;
    uint32_t top1Error = 0, top5Error = 0;
    std::vector<DsImage> ds;

    uint32_t cnt = 0;
    for(uint32_t imgIndex = 0; imgIndex <  imgListSize;){

        // Read images in batchs
        unsigned int batchCnt = 0;
        for(; batchCnt < BATCHSIZE && imgIndex < imgListSize; batchCnt++, imgIndex++){
            progBar(imgIndex, imgListSize);
            std::string imgFilePath = DATASETDIR + "val/" + synsets.at( imgList.at(imgIndex).second) + "/" + imgList.at(imgIndex).first;
            assert(fileExists(imgFilePath));
            ds.push_back(DsImage( imgFilePath, inferNet->getInputH(), inferNet->getInputW()));
        }
        cv::Mat input = blobFromDsImages(ds, inferNet->getInputH(), inferNet->getInputW());

        // Do inference
        auto start = std::chrono::high_resolution_clock::now();
        inferNet->doInference(input.data, batchCnt, prob);
        auto end = std::chrono::high_resolution_clock::now();
        diff += (end - start);
        ds.clear();

        // Calculate the result
        for(unsigned int i = 0; i < BATCHSIZE && cnt < imgListSize; i++){
            float top1 = -2147483647, top2 = -2147483647, top3 = -2147483647, top4 = -2147483647, top5 = -2147483647;
            uint32_t top1Class = 0, top2Class = 0, top3Class = 0, top4Class = 0, top5Class = 0;
            for(unsigned int j = 0; j < inferNet->getOutputSize(); j++){
                float cur = prob[i * (inferNet->getOutputSize()) + j];
                uint32_t curClass = j;
                if(cur > top1) {    std::swap(cur, top1);   std::swap(curClass, top1Class);}
                if(cur > top2) {    std::swap(cur, top2);   std::swap(curClass, top2Class);}
                if(cur > top3) {    std::swap(cur, top3);   std::swap(curClass, top3Class);}
                if(cur > top4) {    std::swap(cur, top4);   std::swap(curClass, top4Class);}
                if(cur > top5) {    std::swap(cur, top5);   std::swap(curClass, top5Class);}
            }

            // Check if top1 and top5 hit the groundtruth
            int noTop1 = 1, noTop5 = 1;
            if(top1Class == imgList[cnt].second) noTop1 = 0, noTop5 = 0;
            if(top2Class == imgList[cnt].second) noTop5 = 0;
            if(top3Class == imgList[cnt].second) noTop5 = 0;
            if(top4Class == imgList[cnt].second) noTop5 = 0;
            if(top5Class == imgList[cnt].second) noTop5 = 0;
            top1Error += noTop1, top5Error += noTop5;
            cnt++;
        }
    }
    std::cout<<"\nOutputs:"<<std::endl;
    std::cout<<"\nBatch size: "<< BATCHSIZE <<std::endl;
    std::cout<<"Top 1 Error rate: "<< 100 * (double)top1Error/imgListSize << "%" << std::endl;
    std::cout<<"Top 5 Error rate: "<< 100 * (double)top5Error/imgListSize << "%" << std::endl;
    std::cout<<"Average inference time over " << imgListSize <<" images: " << 1000 * diff.count()/imgListSize << "ms\n";

    return EXIT_SUCCESS;
}

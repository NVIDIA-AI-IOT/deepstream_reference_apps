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

#include "trt_utils.h"

#include <experimental/filesystem>
#include <fstream>

cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages, const int& inputH,
                         const int& inputW)
{
    std::vector<cv::Mat> inputDataStack(inputImages.size());
    for (uint i = 0; i < inputImages.size(); ++i)
    {
        inputImages.at(i).getProcessedImage().copyTo(inputDataStack.at(i));
    }

    return cv::dnn::blobFromImages(inputDataStack, 1.0, cv::Size(inputW, inputH),
                                   cv::Scalar( 0, 0, 0), false, false);
}


bool fileExists(const std::string fileName)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        std::cout << "File does not exist : " << fileName << std::endl;
        
        return false;
    }

    return true;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"./"};

    return locateFile(input, dirs);
}

std::vector<std::pair<std::string, uint32_t>> loadImageList( std::string fileName){
    std::ifstream infile(fileName);
    std::vector<std::pair<std::string, uint32_t>> imgList;
    if(!infile){
        std::cout<< "Error: Cannot find image list file from " << fileName << ". No such file." <<std::endl;

        return imgList;
    }
    std::string line;
    while(getline(infile, line)){
        std::istringstream iss(line);
        std::string imgName;
        uint32_t imgLabel;
        if(!(iss >> imgName >> std::dec >> imgLabel)){    break;}
        imgList.push_back({imgName, imgLabel});
    }

    return imgList;
}

std::vector<std::string> getSynsets( std::string fileName){
    std::ifstream infile(fileName);
    std::vector<std::string> synsets;
    if(!infile){
        std::cout<< "Error: Cannot find synsets file from " << fileName << ". No such file." <<std::endl;

        return synsets;
    }
    std::string line;
    while(getline(infile, line)){
        synsets.push_back(line);
    }

    return synsets;
}

nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, nvinfer1::ILogger& logger)
{
    // reading the model in memory
    std::cout << "Loading TRT Engine..." << std::endl;
    assert(fileExists(planFilePath));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(planFilePath);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculating model size
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine
        = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);
    free(modelMem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input>> std::dec >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<nvinfer1::DataType>(type);
        // Load blob
        if (wt.type == nvinfer1::DataType::kFLOAT)
        {
            float* val = reinterpret_cast<float*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> val[x];
            }
            wt.values = val;
        }
        else if (wt.type == nvinfer1::DataType::kHALF)
        {
            uint16_t* val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

int getWidth(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[1];
}

int getHight(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[2];
}

void progBar(int index, int maximum){
    int tmp = (float)index/maximum * 69; 
    float cnt = (float)index/maximum * 100;
    for(int progbar = 0; progbar < 75; progbar++)
    {
        if(progbar == 0) 
            std::cout<<"|";
        else if(progbar == 69)
        {
            if (cnt > 99.5)
                    std::cout<< "| 100%\r";   
            else
                    std::cout<< "| " << (int)cnt << "%\r";
        }
        else if(progbar<=tmp)
            std::cout<< (char)2;
        else if(progbar<69)
            std::cout<<" ";
    }
}

nvinfer1::IScaleLayer* addBN(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, std::map<std::string, nvinfer1::Weights>& weightMap, const std::string name){
    std::string beta = name + "_beta", gamma = name + "_gamma", mvm = name + "_mean_EMA", mvv = name + "_variance_EMA";
    nvinfer1::DataType dt = weightMap[name].type;
    // create the weights
    nvinfer1::Weights shift{dt, nullptr, weightMap[beta].count};
    nvinfer1::Weights scale{dt, nullptr, weightMap[beta].count};
    nvinfer1::Weights power{dt, nullptr, weightMap[beta].count};
    const float* beta_ptr = (const float*)weightMap[beta].values;
    const float* gamma_ptr = (const float*)weightMap[gamma].values;
    const float* mvm_ptr = (const float*)weightMap[mvm].values;
    const float* mvv_ptr = (const float*)weightMap[mvv].values;
    float* shiftWt = new float[weightMap[beta].count];
    float* scaleWt = new float[weightMap[beta].count];
    float* powerWt = new float[weightMap[beta].count];
    
    for (int i = 0; i < weightMap[beta].count; ++i)
    {
        float stdv = sqrt(mvv_ptr[i] + 1.0e-5);
        shiftWt[i] = (beta_ptr[i]) - (((mvm_ptr[i]) * (gamma_ptr[i])) / stdv);
        scaleWt[i] = (gamma_ptr[i]) / stdv;
        powerWt[i] = 1.0;
    }
    shift.values = shiftWt;
    scale.values = scaleWt;
    power.values = powerWt;
    nvinfer1::IScaleLayer* bn = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn);

    return bn;
}
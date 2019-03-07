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

#include "se_resnet50.h"

SE_ResNet50::SE_ResNet50() :
    m_Precision(config::kPRECISION),
    m_BatchSize(config::kBATCHSIZE),
    m_InputC(config::kINPUT_C),
    m_InputH(config::kINPUT_H),
    m_InputW(config::kINPUT_W),
    m_OutputSize(config::kOUTPUTSIZE),
    inputIndex(-1),
    outputIndex(-1),
    m_InputBlobName(config::kINPUT_BLOB_NAME),
    m_OutputBlobName(config::kOUTPUT_BLOB_NAME),
    m_PlanFilePath(config::kPLAN_FILE_PATH),
    m_CalibImagesFileDir(config::kIMAGE_DATASET_DIR),
    m_CalibTableFilePath(config::kCALIB_TABLE_PATH),
    m_WeightFilePath(config::kTRAINED_WEIGHTS_PATH),
    m_MaxWorkSpaceSize(config::kMAXWORKSPACESIZE)
{
    if (!fileExists(m_PlanFilePath))
    {
        std::cout << "Unable to find cached TensorRT engine for network : SE-ResNet50-"
                  << "precision-" << m_Precision << "-batch_size-" << std::to_string(m_BatchSize)
                  << std::endl;
        std::cout << "Creating a new TensorRT Engine" << std::endl;

        if(m_Precision == "kFLOAT")
        {
            createSE_Resnet50Engine(m_BatchSize, DataType::kFLOAT, m_WeightFilePath, nullptr);
        }
        else if (m_Precision == "kHALF")
        {
            createSE_Resnet50Engine(m_BatchSize, DataType::kHALF, m_WeightFilePath, nullptr);
        }
        else if (m_Precision == "kINT8")
        {
            Int8EntropyCalibrator calibrator(m_BatchSize, m_CalibImagesFileDir,
                                             m_CalibTableFilePath, m_InputC * m_InputH * m_InputW, m_InputH, m_InputW,
                                             m_InputBlobName);
            createSE_Resnet50Engine(m_BatchSize, DataType::kINT8, m_WeightFilePath, &calibrator);
        }
        else
        {
            std::cout << "Unrecognized precision type " << m_Precision << std::endl;
            assert(0);
        }
    }
    else
    {
        std::cout << "Using previously generated plan file located at " << m_PlanFilePath
                  << std::endl;
    }

    ICudaEngine* engine = loadTRTEngine(m_PlanFilePath, gLogger);
    assert(engine != nullptr);
    m_context = engine->createExecutionContext();
    assert(m_context != nullptr);
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(m_InputBlobName.c_str());
    outputIndex = engine->getBindingIndex(m_OutputBlobName.c_str());

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], m_BatchSize * m_InputC * m_InputH * m_InputW * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], m_BatchSize * m_OutputSize * sizeof(float)));

    // Create stream
    CHECK(cudaStreamCreate(&m_stream));
};

SE_ResNet50::~SE_ResNet50()
{
    // Release stream and buffers
    cudaStreamDestroy(m_stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void SE_ResNet50::createSE_Resnet50Engine(const unsigned int maxBatchSize,
                                    const DataType dataType,
                                    const std::string trainedWeightsPath,
                                    nvinfer1::IInt8EntropyCalibrator* calibrator)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
    {
        std::cout << "Platform doesn't support this precision." << std::endl;
        assert(0);
    }

    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape { 1, 3, 224, 224 } with name m_InputBlobName
    ITensor* data = network->addInput(m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT, Dims3{3, m_InputH, m_InputW});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(locateFile(trainedWeightsPath));

    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    // Add asymmetrical padding
    IPaddingLayer* conv0_pad = network->addPadding(*data, DimsHW{2, 2}, DimsHW{3, 3});

    IConvolutionLayer* conv = network->addConvolution(*conv0_pad->getOutput(0), 64, DimsHW{7, 7}, weightMap["conv0_W"], convBias);
    assert(conv);
    conv->setStride(DimsHW{2, 2});
    conv->setName("conv0");

    // Add batch normalization layer
    IScaleLayer* bn = addBN(network, *conv->getOutput(0), weightMap, "conv0_bn");
    assert(bn);
    bn->setName("conv0_bn");

    // Add relu
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu);
    relu->setName("conv0_relu");

    // Add asymmetrical padding
    IPaddingLayer* pool0_maxPool_pad = network->addPadding(*relu->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});

    // Add max pooling layer with stride of 2x2 and kernel size of 3x3.
    IPoolingLayer* pool0_maxPool = network->addPooling(*pool0_maxPool_pad->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool0_maxPool);
    pool0_maxPool->setStride(DimsHW{2, 2});
    pool0_maxPool->setName("pool0_maxPool");

    std::vector<int> blocks = {3, 4, 6, 3};
    std::vector<int> kernel = {1, 3, 1};
    std::vector<int> pad = {0, 1, 0};
    std::vector<int> feature = {1, 1, 4};
    nvinfer1::ITensor* previous_left = pool0_maxPool->getOutput(0);
    int32_t feature_space = 64;
    int32_t global_pool_dim = 56;
    for(int group = 0; group<4; group++){
        for(int block = 0; block<blocks[group]; block++){
            nvinfer1::ITensor* previous_right = previous_left;
            if(block==0){
                std::string conv_name = "group" + to_string(group) + "_block0_convshortcut_W";
                std::string batch_name = "group" + to_string(group) + "_block0_convshortcut_bn";
                // 1x1 convolution
                IConvolutionLayer* conv1 = network->addConvolution(*previous_left,
                                                                    feature_space * 4,
                                                                    DimsHW{1, 1},
                                                                    weightMap[conv_name],
                                                                    convBias);
                assert(conv1);
                conv1->setName(("group" + to_string(group) + "_block0_convshortcut").c_str());
                if(group!=0){
                    conv1->setStride(DimsHW{2, 2});}
                // Add batch normalization layer
                IScaleLayer* bn1= addBN(network, *conv1->getOutput(0), weightMap, batch_name);
                bn1->setName(("group" + to_string(group) + "_block0_convshortcut_bn").c_str());

                previous_left = bn1->getOutput(0);
            }
            for(int convBnRelu = 0; convBnRelu <3; convBnRelu++){
                // Add asymmetrical padding
                IPaddingLayer* conv2_pad = network->addPadding(*previous_right, DimsHW{pad[convBnRelu], pad[convBnRelu]}, DimsHW{pad[convBnRelu], pad[convBnRelu]});

                if(group!=0 && block==0 && convBnRelu == 1)
                {
                    conv2_pad->setPrePadding(DimsHW{0, 0});
                }

                std::string conv_name = "group" + to_string(group) + "_block" + to_string(block) + "_conv" + to_string(convBnRelu+1) + "_W";
                std::string batch_name = "group" + to_string(group) + "_block" + to_string(block) + "_conv" + to_string(convBnRelu+1) + "_bn";
                // 1x1 convolution
                IConvolutionLayer* conv2 = network->addConvolution(*conv2_pad->getOutput(0),
                                                                   feature_space * feature[convBnRelu],
                                                                   DimsHW{kernel[convBnRelu], kernel[convBnRelu]},
                                                                   weightMap[conv_name],
                                                                   convBias);
                assert(conv2);
                conv2->setName(("group" + to_string(group) + "_block" + to_string(block) + "_conv" + to_string(convBnRelu+1)).c_str());

                if(group!=0 && block==0 && convBnRelu == 1){
                    conv2->setStride(DimsHW{2, 2});}

                // Add batch normalization layer
                IScaleLayer* bn2 = addBN(network, *conv2->getOutput(0), weightMap, batch_name);
                assert(bn2);
                bn2->setName(("group" + to_string(group) + "_block" + to_string(block) + "_conv" + to_string(convBnRelu+1) + "_bn").c_str());

                // Add relu
                if(convBnRelu!=2){
                    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
                    assert(relu2);
                    relu2->setName(("group" + to_string(group) + "_block" + to_string(block) + "_conv" + to_string(convBnRelu+1) + "_relu").c_str());
                    previous_right = relu2->getOutput(0);
                }
                else{
                    // Add SE block
                    IPoolingLayer* global_pool = network->addPooling(*bn2->getOutput(0), PoolingType::kAVERAGE, DimsHW{global_pool_dim, global_pool_dim});
                    assert(global_pool);
                    global_pool->setName(("group" + to_string(group) + "_block" + to_string(block) + "_global_pooling").c_str());

                    std::string fc_name = "group" + to_string(group) + "_block" + to_string(block) + "_fc";
                    IFullyConnectedLayer* fc1 = network->addFullyConnected(*global_pool->getOutput(0), feature_space / 4 , weightMap[fc_name+"1_W"], weightMap[fc_name+"1_b"]);
                    assert(fc1);
                    fc1->setName((fc_name + "1").c_str());

                    IActivationLayer* relu2 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
                    assert(relu2);
                    relu2->setName(("group" + to_string(group) + "_block" + to_string(block) + "_fc1_relu").c_str());

                    IFullyConnectedLayer* fc2 = network->addFullyConnected(*relu2->getOutput(0), feature_space * 4, weightMap[fc_name+"2_W"], weightMap[fc_name+"2_b"]);
                    assert(fc2);
                    fc2->setName((fc_name + "2").c_str());

                    IActivationLayer* sigmoid = network->addActivation(*fc2->getOutput(0), ActivationType::kSIGMOID);
                    assert(sigmoid);
                    sigmoid->setName(("group" + to_string(group) + "_block" + to_string(block) + "_fc1_sigmoid").c_str());

                    IElementWiseLayer* mul = network->addElementWise(*sigmoid->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kPROD);
                    assert(mul);
                    mul->setName(("group" + to_string(group) + "_block" + to_string(block) + "_mul").c_str());
                    previous_right = mul->getOutput(0);
                }
            }
            IElementWiseLayer* add = network->addElementWise(*previous_left, *previous_right, ElementWiseOperation::kSUM);
            assert(add);
            add->setName(("group" + to_string(group) + "_block" + to_string(block) + "_add").c_str());
            IActivationLayer* relu3 = network->addActivation(*add->getOutput(0), ActivationType::kRELU);
            assert(relu3);
            relu3->setName(("group" + to_string(group) + "_block" + to_string(block) + "_relu").c_str());
            previous_left = relu3->getOutput(0);
        }
        feature_space *= 2;
        global_pool_dim /= 2;
    }

    IPoolingLayer* gap_meanPool = network->addPooling(*previous_left, PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(gap_meanPool);
    gap_meanPool->setName("gap_meanPool");

    // Add fully connected layer with 1000 outputs.
    IFullyConnectedLayer* linear = network->addFullyConnected(*gap_meanPool->getOutput(0), 1000, weightMap["linear_W"], weightMap["linear_b"]);
    assert(linear);
    linear->setName("linear");

    linear->getOutput(0)->setName(m_OutputBlobName.c_str());
    network->markOutput(*linear->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(m_MaxWorkSpaceSize);

    if (dataType == nvinfer1::DataType::kHALF)
    {
        builder->setHalf2Mode(true);
    }
    else if (dataType == nvinfer1::DataType::kINT8)
    {
        assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);
    std::cout << "Building complete!" << std::endl;

    // Serialize the engine
    std::cout << "Serializing the TensorRT Engine..." << std::endl;
    nvinfer1::IHostMemory* modelStream = engine->serialize();
    assert(modelStream != nullptr);

    // write data to output file
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write(static_cast<const char*>(modelStream->data()), modelStream->size());
    std::ofstream outFile;
    outFile.open(m_PlanFilePath);
    outFile << gieModelStream.rdbuf();
    outFile.close();
    std::cout << "Serialized plan file cached at location : " << m_PlanFilePath << std::endl;

    // Don't need the network any more
    network->destroy();
    engine->destroy();
    builder->destroy();
    std::cout<<"network destroyed."<<std::endl;
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }
    std::cout<<"memory released."<<std::endl;
}

void SE_ResNet50::doInference(const unsigned char* input, const int batchSize, float* output)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * m_InputC * m_InputH * m_InputW * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    m_context->enqueue(batchSize, buffers, m_stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * m_OutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    cudaStreamSynchronize(m_stream);
}

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

#include "yolo.h"
#include "network_config.h"

Yolo::Yolo(uint batchSize) :
    m_ModelsPath(config::kMODELS_PATH),
    m_ConfigFilePath(config::kYOLO_CONFIG_PATH),
    m_TrainedWeightsPath(config::kTRAINED_WEIGHTS_PATH),
    m_NetworkType(config::kNETWORK_TYPE),
    m_CalibImagesFilePath(config::kCALIBRATION_SET),
    m_CalibTableFilePath(config::kCALIB_TABLE_PATH),
    m_Precision(config::kPRECISION),
    m_InputBlobName(config::kINPUT_BLOB_NAME),
    m_InputH(config::kINPUT_H),
    m_InputW(config::kINPUT_W),
    m_InputC(config::kINPUT_C),
    m_InputSize(config::kINPUT_SIZE),
    m_NumOutputClasses(config::kOUTPUT_CLASSES),
    m_NumBBoxes(config::kBBOXES),
    m_ProbThresh(config::kPROB_THRESH),
    m_NMSThresh(config::kNMS_THRESH),
    m_Anchors(config::kANCHORS),
    m_ClassNames(config::kCLASS_NAMES),
    m_PrintPerfInfo(config::kPRINT_PERF_INFO),
    m_PrintPredictions(config::kPRINT_PRED_INFO),
    m_Logger(Logger()),
    m_BatchSize(batchSize),
    m_Engine(nullptr),
    m_Context(nullptr),
    m_Bindings(),
    m_TrtOutputBuffers(),
    m_InputIndex(-1),
    m_CudaStream(nullptr),
    m_PluginFactory(new PluginFactory),
    m_TinyMaxpoolPaddingFormula(new YoloTinyMaxpoolPaddingFormula)
{
    std::string planFilePath = m_ModelsPath + m_NetworkType + "-" + m_Precision + "-batch"
        + std::to_string(m_BatchSize) + ".engine";
    // Create and cache the engine if not already present
    if (!fileExists(planFilePath))
    {
        std::cout << "Unable to find cached TensorRT engine for network : " << m_NetworkType
                  << " precision : " << m_Precision << " and batch size :" << m_BatchSize
                  << std::endl;
        std::cout << "Creating a new TensorRT Engine" << std::endl;

        if (m_Precision == "kFLOAT")
        {
            createYOLOEngine(m_BatchSize, m_ConfigFilePath, m_TrainedWeightsPath, planFilePath);
        }
        else if (m_Precision == "kINT8")
        {
            Int8EntropyCalibrator calibrator(m_BatchSize, m_CalibImagesFilePath,
                                             m_CalibTableFilePath, m_InputSize, m_InputH, m_InputW,
                                             m_InputBlobName);
            createYOLOEngine(m_BatchSize, m_ConfigFilePath, m_TrainedWeightsPath, planFilePath,
                             nvinfer1::DataType::kINT8, &calibrator);
        }
        else if (m_Precision == "kHALF")
        {
            createYOLOEngine(m_BatchSize, m_ConfigFilePath, m_TrainedWeightsPath, planFilePath,
                             nvinfer1::DataType::kHALF, nullptr);
        }
        else
        {
            std::cout << "Unrecognized precision type " << m_Precision << std::endl;
            assert(0);
        }
    }
    else
        std::cout << "Using previously generated plan file located at " << planFilePath
                  << std::endl;

    assert(m_PluginFactory != nullptr);
    m_Engine = loadTRTEngine(planFilePath, m_PluginFactory, m_Logger);
    assert(m_Engine != nullptr);
    m_Context = m_Engine->createExecutionContext();
    assert(m_Context != nullptr);
    m_Bindings.resize(m_Engine->getNbBindings(), nullptr);
    m_TrtOutputBuffers.resize(m_Engine->getNbBindings() - 1, nullptr);
    m_InputIndex = m_Engine->getBindingIndex(m_InputBlobName.c_str());
    assert(m_InputIndex != -1);
    assert(m_BatchSize <= static_cast<uint>(m_Engine->getMaxBatchSize()));
    NV_CUDA_CHECK(cudaStreamCreate(&m_CudaStream));
};

Yolo::~Yolo()
{
    for (auto buffer : m_TrtOutputBuffers) delete[] buffer;
    for (auto binding : m_Bindings) NV_CUDA_CHECK(cudaFree(binding));
    cudaStreamDestroy(m_CudaStream);
    if (m_Context)
    {
        m_Context->destroy();
        m_Context = nullptr;
    }

    if (m_Engine)
    {
        m_Engine->destroy();
        m_Engine = nullptr;
    }

    m_PluginFactory->destroy();
}

void Yolo::createYOLOEngine(const int batchSize, const std::string yoloConfigPath,
                            const std::string trainedWeightsPath, const std::string planFilePath,
                            const nvinfer1::DataType dataType, Int8EntropyCalibrator* calibrator)
{
    assert(fileExists(yoloConfigPath));
    assert(fileExists(trainedWeightsPath));

    std::vector<std::map<std::string, std::string>> blocks = parseConfig(yoloConfigPath);
    std::vector<float> weights = loadWeights(trainedWeightsPath, m_NetworkType);
    std::vector<nvinfer1::Weights> trtWeights;
    int weightPtr = 0;
    int channels = m_InputC;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(m_Logger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
    {
        std::cout << "Platform doesn't support this precision." << std::endl;
        assert(0);
    }

    nvinfer1::ITensor* data = network->addInput(m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
                                                nvinfer1::DimsCHW{static_cast<int>(m_InputC),
                                                                  static_cast<int>(m_InputH),
                                                                  static_cast<int>(m_InputW)});
    nvinfer1::ITensor* previous = data;
    std::vector<nvinfer1::ITensor*> tensorOutputs;
    std::vector<nvinfer1::ITensor*> outputLayers;

    // Set the output dimensions formula for pooling layers
    network->setPoolingOutputDimensionsFormula(m_TinyMaxpoolPaddingFormula.get());

    // build the network using the network API
    for (uint i = 0; i < blocks.size(); ++i)
    {
        // check if num. of channels is correct
        assert(getNumChannels(previous) == channels);
        std::string layerIndex = "(" + std::to_string(i) + ")";

        if (blocks.at(i).at("type") == "net")
        {
            printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
        }
        else if (blocks.at(i).at("type") == "convolutional")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out;
            std::string layerType;
            // check if batch_norm enabled
            if (blocks.at(i).find("batch_normalize") != blocks.at(i).end())
            {
                out = netAddConvBNLeaky(i, blocks.at(i), weights, trtWeights, weightPtr, channels,
                                        previous, network);
                layerType = "conv-bn-leaky";
            }
            else
            {
                out = netAddConvLinear(i, blocks.at(i), weights, trtWeights, weightPtr, channels,
                                       previous, network);
                layerType = "conv-linear";
            }
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
        }
        else if (blocks.at(i).at("type") == "shortcut")
        {
            assert(blocks.at(i).at("activation") == "linear");
            assert(blocks.at(i).find("from") != blocks.at(i).end());
            int from = stoi(blocks.at(i).at("from"));

            std::string inputVol = dimsToString(previous->getDimensions());
            // check if indexes are correct
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);
            nvinfer1::IElementWiseLayer* ew
                = network->addElementWise(*tensorOutputs[i - 2], *tensorOutputs[i + from - 1],
                                          nvinfer1::ElementWiseOperation::kSUM);
            assert(ew != nullptr);
            std::string ewLayerName = "shortcut_" + std::to_string(i);
            ew->setName(ewLayerName.c_str());
            previous = ew->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(ew->getOutput(0));
            printLayerInfo(layerIndex, "skip", inputVol, outputVol, "    -");
        }
        else if (blocks.at(i).at("type") == "yolo")
        {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            uint gridSize = prevTensorDims.d[1];
            nvinfer1::IPlugin* yoloPlugin
                = new YoloLayerV3(m_NumBBoxes, m_NumOutputClasses, gridSize);
            assert(yoloPlugin != nullptr);
            nvinfer1::IPluginLayer* yolo = network->addPlugin(&previous, 1, *yoloPlugin);
            assert(yolo != nullptr);
            std::string layerName = "yolo_" + std::to_string(i);
            yolo->setName(layerName.c_str());
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = yolo->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network->markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(yolo->getOutput(0));
            outputLayers.push_back(tensorOutputs.back());
            printLayerInfo(layerIndex, "yolo", inputVol, outputVol, std::to_string(weightPtr));
        }
        else if (blocks.at(i).at("type") == "region")
        {
            nvinfer1::plugin::RegionParameters RegionParameters{
                static_cast<int>(m_NumBBoxes), 4, static_cast<int>(m_NumOutputClasses), nullptr};
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPlugin* regionPlugin
                = nvinfer1::plugin::createYOLORegionPlugin(RegionParameters);
            assert(regionPlugin != nullptr);
            nvinfer1::IPluginLayer* region = network->addPlugin(&previous, 1, *regionPlugin);
            assert(region != nullptr);

            std::string layerName = "region_" + std::to_string(i);
            region->setName(layerName.c_str());

            previous = region->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            network->markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(region->getOutput(0));
            outputLayers.push_back(tensorOutputs.back());
            printLayerInfo(layerIndex, "region", inputVol, outputVol, std::to_string(weightPtr));
        }
        else if (blocks.at(i).at("type") == "reorg")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::IPlugin* reorgPlugin = nvinfer1::plugin::createYOLOReorgPlugin(2);
            assert(reorgPlugin != nullptr);
            nvinfer1::IPluginLayer* reorg = network->addPlugin(&previous, 1, *reorgPlugin);
            assert(reorg != nullptr);

            std::string layerName = "reorg_" + std::to_string(i);
            reorg->setName(layerName.c_str());
            previous = reorg->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            channels = getNumChannels(previous);
            tensorOutputs.push_back(reorg->getOutput(0));
            printLayerInfo(layerIndex, "reorg", inputVol, outputVol, std::to_string(weightPtr));
        }
        // route layers (single or concat)
        else if (blocks.at(i).at("type") == "route")
        {
            size_t found = blocks.at(i).at("layers").find(",");
            if (found != std::string::npos)
            {
                int idx1 = std::stoi(trim(blocks.at(i).at("layers").substr(0, found)));
                int idx2 = std::stoi(trim(blocks.at(i).at("layers").substr(found + 1)));
                if (idx1 < 0)
                {
                    idx1 = tensorOutputs.size() + idx1;
                }
                if (idx2 < 0)
                {
                    idx2 = tensorOutputs.size() + idx2;
                }
                assert(idx1 < static_cast<int>(tensorOutputs.size()) && idx1 >= 0);
                assert(idx2 < static_cast<int>(tensorOutputs.size()) && idx2 >= 0);
                nvinfer1::ITensor** concatInputs
                    = reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * 2));
                concatInputs[0] = tensorOutputs[idx1];
                concatInputs[1] = tensorOutputs[idx2];
                nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs, 2);
                assert(concat != nullptr);
                std::string concatLayerName = "route_" + std::to_string(i - 1);
                concat->setName(concatLayerName.c_str());
                // concatenate along the channel dimension
                concat->setAxis(0);
                previous = concat->getOutput(0);
                assert(previous != nullptr);
                std::string outputVol = dimsToString(previous->getDimensions());
                // set the output volume depth
                channels
                    = getNumChannels(tensorOutputs[idx1]) + getNumChannels(tensorOutputs[idx2]);
                tensorOutputs.push_back(concat->getOutput(0));
                printLayerInfo(layerIndex, "route", "        -", outputVol,
                               std::to_string(weightPtr));
            }
            else
            {
                int idx = std::stoi(trim(blocks.at(i).at("layers")));
                if (idx < 0)
                {
                    idx = tensorOutputs.size() + idx;
                }
                assert(idx < static_cast<int>(tensorOutputs.size()) && idx >= 0);
                previous = tensorOutputs[idx];
                assert(previous != nullptr);
                std::string outputVol = dimsToString(previous->getDimensions());
                // set the output volume depth
                channels = getNumChannels(tensorOutputs[idx]);
                tensorOutputs.push_back(tensorOutputs[idx]);
                printLayerInfo(layerIndex, "route", "        -", outputVol,
                               std::to_string(weightPtr));
            }
        }
        else if (blocks.at(i).at("type") == "upsample")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out
                = netAddUpsample(i - 1, blocks[i], weights, channels, previous, network);
            previous = out->getOutput(0);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        }
        else if (blocks.at(i).at("type") == "maxpool")
        {
            // Add same padding layers
            if (blocks.at(i).at("size") == "2" && blocks.at(i).at("stride") == "1")
            {
                m_TinyMaxpoolPaddingFormula->addSamePaddingLayer("maxpool_" + std::to_string(i));
            }
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddMaxpool(i, blocks.at(i), previous, network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        }
        else
        {
            std::cout << "Unsupported layer type --> \"" << blocks.at(i).at("type") << "\""
                      << std::endl;
            assert(0);
        }
    }

    if (weights.size() != weightPtr)
    {
        std::cout << "Number of unused weights left : " << weights.size() - weightPtr << std::endl;
        assert(0);
    }

    std::cout << "Output layers :" << std::endl;
    for (auto layer : outputLayers) std::cout << layer->getName() << std::endl;

    builder->setMaxBatchSize(batchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    if (dataType == nvinfer1::DataType::kINT8)
    {
        assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }
    else if (dataType == nvinfer1::DataType::kHALF)
    {
        builder->setHalf2Mode(true);
    }

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
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
    outFile.open(planFilePath);
    outFile << gieModelStream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << planFilePath << std::endl;
    network->destroy();
    engine->destroy();
    builder->destroy();
    modelStream->destroy();

    // deallocate the weights
    for (uint i = 0; i < trtWeights.size(); ++i)
    {
        free(const_cast<void*>(trtWeights[i].values));
    }
}
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

#ifndef __LEGACY_CAFFE_PLUGIN_FACTORY__
#define __LEGACY_CAFFE_PLUGIN_FACTORY__

#include <cassert>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <iostream>
#include <memory>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "fp16.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)                                                                              \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << status;                                               \
            abort();                                                                               \
        }                                                                                          \
    }

class FCPlugin : public IPluginExt
{
public:
    FCPlugin(const Weights* weights, int nbWeights, int nbOutputChannels) :
        mNbOutputChannels(nbOutputChannels)
    {
        assert(nbWeights == 2);

        mKernelWeights = weights[0];
        assert(mKernelWeights.type == DataType::kFLOAT || mKernelWeights.type == DataType::kHALF);

        mBiasWeights = weights[1];
        assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);
        assert(mBiasWeights.type == DataType::kFLOAT || mBiasWeights.type == DataType::kHALF);

        mKernelWeights.values = malloc(mKernelWeights.count * type2size(mKernelWeights.type));
        memcpy(const_cast<void*>(mKernelWeights.values), weights[0].values,
               mKernelWeights.count * type2size(mKernelWeights.type));
        mBiasWeights.values = malloc(mBiasWeights.count * type2size(mBiasWeights.type));
        memcpy(const_cast<void*>(mBiasWeights.values), weights[1].values,
               mBiasWeights.count * type2size(mBiasWeights.type));

        mNbInputChannels = int(weights[0].count / nbOutputChannels);
    }

    // create the plugin at runtime from a byte stream
    FCPlugin(const void* data, size_t length)
    {
        const char *d = static_cast<const char*>(data), *a = d;
        read(d, mNbInputChannels);
        read(d, mNbOutputChannels);

        mKernelWeights.count = mNbInputChannels * mNbOutputChannels;
        mKernelWeights.values = nullptr;

        read(d, mBiasWeights.count);
        mBiasWeights.values = nullptr;

        read(d, mDataType);

        deserializeToDevice(d, mDeviceKernel, mKernelWeights.count * type2size(mDataType));
        deserializeToDevice(d, mDeviceBias, mBiasWeights.count * type2size(mDataType));
        assert(d == a + length);
    }

    ~FCPlugin()
    {
        if (mKernelWeights.values)
        {
            free(const_cast<void*>(mKernelWeights.values));
            mKernelWeights.values = nullptr;
        }
        if (mBiasWeights.values)
        {
            free(const_cast<void*>(mBiasWeights.values));
            mBiasWeights.values = nullptr;
        }
    }

    int getNbOutputs() const override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
        return Dims3(mNbOutputChannels, 1, 1);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT || type == DataType::kHALF)
            && format == PluginFormat::kNCHW;
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                             int nbOutputs, DataType type, PluginFormat format,
                             int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT || type == DataType::kHALF)
               && format == PluginFormat::kNCHW);
        mDataType = type;
    }

    int initialize() override
    {
        CHECK(cudnnCreate(&mCudnn)); // initialize cudnn and cublas
        CHECK(cublasCreate(&mCublas));
        CHECK(cudnnCreateTensorDescriptor(
            &mSrcDescriptor)); // create cudnn tensor descriptors we need for bias addition
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        if (mKernelWeights.values) convertAndCopyToDevice(mDeviceKernel, mKernelWeights);
        if (mBiasWeights.values) convertAndCopyToDevice(mDeviceBias, mBiasWeights);

        return 0;
    }

    virtual void terminate() override
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cublasDestroy(mCublas));
        CHECK(cudnnDestroy(mCudnn));
        if (mDeviceKernel)
        {
            cudaFree(mDeviceKernel);
            mDeviceKernel = nullptr;
        }
        if (mDeviceBias)
        {
            cudaFree(mDeviceBias);
            mDeviceBias = nullptr;
        }
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
                        cudaStream_t stream) override
    {
        float onef{1.0f}, zerof{0.0f};
        __half oneh = fp16::__float2half(1.0f), zeroh = fp16::__float2half(0.0f);

        cublasSetStream(mCublas, stream);
        cudnnSetStream(mCudnn, stream);

        if (mDataType == DataType::kFLOAT)
        {
            CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize,
                              mNbInputChannels, &onef,
                              reinterpret_cast<const float*>(mDeviceKernel), mNbInputChannels,
                              reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &zerof,
                              reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
        }
        else
        {
            CHECK(cublasHgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize,
                              mNbInputChannels, &oneh,
                              reinterpret_cast<const __half*>(mDeviceKernel), mNbInputChannels,
                              reinterpret_cast<const __half*>(inputs[0]), mNbInputChannels, &zeroh,
                              reinterpret_cast<__half*>(outputs[0]), mNbOutputChannels));
        }
        if (mBiasWeights.count)
        {
            cudnnDataType_t cudnnDT
                = mDataType == DataType::kFLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
            CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, cudnnDT, 1,
                                             mNbOutputChannels, 1, 1));
            CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, cudnnDT, batchSize,
                                             mNbOutputChannels, 1, 1));
            CHECK(cudnnAddTensor(mCudnn, &onef, mSrcDescriptor, mDeviceBias, &onef, mDstDescriptor,
                                 outputs[0]));
        }

        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        return sizeof(mNbInputChannels) + sizeof(mNbOutputChannels) + sizeof(mBiasWeights.count)
            + sizeof(mDataType)
            + (mKernelWeights.count + mBiasWeights.count) * type2size(mDataType);
    }

    virtual void serialize(void* buffer) override
    {
        char *d = static_cast<char*>(buffer), *a = d;

        write(d, mNbInputChannels);
        write(d, mNbOutputChannels);
        write(d, mBiasWeights.count);
        write(d, mDataType);
        convertAndCopyToBuffer(d, mKernelWeights);
        convertAndCopyToBuffer(d, mBiasWeights);
        assert(d == a + getSerializationSize());
    }

private:
    size_t type2size(DataType type)
    {
        return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half);
    }

    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count));
        CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
        return deviceData;
    }

    void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights)
    {
        if (weights.type
            != mDataType) // Weights are converted in host memory first, if the type does not match
        {
            size_t size
                = weights.count * (mDataType == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
            void* buffer = malloc(size);
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    static_cast<float*>(buffer)[v]
                        = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    static_cast<__half*>(buffer)[v]
                        = fp16::__float2half(static_cast<const float*>(weights.values)[v]);

            deviceWeights = copyToDevice(buffer, size);
            free(buffer);
        }
        else
            deviceWeights = copyToDevice(weights.values, weights.count * type2size(mDataType));
    }

    void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
    {
        if (weights.type != mDataType)
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    reinterpret_cast<float*>(buffer)[v]
                        = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    reinterpret_cast<__half*>(buffer)[v]
                        = fp16::__float2half(static_cast<const float*>(weights.values)[v]);
        else
            memcpy(buffer, weights.values, weights.count * type2size(mDataType));
        buffer += weights.count * type2size(mDataType);
    }

    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    int mNbOutputChannels, mNbInputChannels;
    Weights mKernelWeights, mBiasWeights;

    DataType mDataType{DataType::kFLOAT};
    void* mDeviceKernel{nullptr};
    void* mDeviceBias{nullptr};

    cudnnHandle_t mCudnn;
    cublasHandle_t mCublas;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

// integration for serialization
class CaffeMNISTPluginFactoryLegacy : public nvinfer1::IPluginFactory,
                                      public nvcaffeparser1::IPluginFactoryExt
{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override { return isPluginExt(name); }

    bool isPluginExt(const char* name) override { return !strcmp(name, "ip2"); }

    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights,
                                            int nbWeights) override
    {
        // there's no way to pass parameters through from the model definition, so we have to define
        // it here explicitly
        static const int NB_OUTPUT_CHANNELS = 10;
        assert(isPlugin(layerName) && nbWeights == 2);
        assert(mPlugin.get() == nullptr);
        mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(weights, nbWeights, NB_OUTPUT_CHANNELS));
        return mPlugin.get();
    }

    // deserialization plugin implementation
    IPlugin* createPlugin(const char* layerName, const void* serialData,
                          size_t serialLength) override
    {
        std::cout << "Calling within plugin factory" << std::endl;
        assert(isPlugin(layerName));
        assert(mPlugin.get() == nullptr);
        mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(serialData, serialLength));
        return mPlugin.get();
    }

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin() { mPlugin.reset(); }

    std::unique_ptr<FCPlugin> mPlugin{nullptr};
};

#endif // __LEGACY_CAFFE_PLUGIN_FACTORY__
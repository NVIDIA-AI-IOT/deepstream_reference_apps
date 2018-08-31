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

#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <cassert>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <memory>

#include "NvInferPlugin.h"

#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ \
                      << std::endl;                                                                \
            abort();                                                                               \
        }                                                                                          \
    }

class PluginFactory : public nvinfer1::IPluginFactory
{

public:
    PluginFactory();
    nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData,
                                    size_t serialLength) override;
    bool isPlugin(const char* name);
    void destroy();

private:
    static const int m_MaxLeakyLayers = 72;
    static const int m_ReorgStride = 2;
    static constexpr float m_LeakyNegSlope = 0.1;
    static const int m_NumBoxes = 5;
    static const int m_NumCoords = 4;
    static const int m_NumClasses = 80;
    int m_LeakyReLUCount = 0;
    nvinfer1::plugin::RegionParameters m_RegionParameters{m_NumBoxes, m_NumCoords, m_NumClasses,
                                                          nullptr};

    struct nvPluginDeleter
    {
        void operator()(nvinfer1::plugin::INvPlugin* ptr)
        {
            if (ptr)
            {
                ptr->destroy();
            }
        }
    };
    typedef std::unique_ptr<nvinfer1::plugin::INvPlugin, nvPluginDeleter> unique_ptr_INvPlugin;

    unique_ptr_INvPlugin m_ReorgLayer;
    unique_ptr_INvPlugin m_RegionLayer;
    unique_ptr_INvPlugin m_LeakyReLULayers[m_MaxLeakyLayers];
};

#endif // __PLUGIN_LAYER_H__
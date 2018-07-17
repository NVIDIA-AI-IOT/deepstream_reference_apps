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

#include "plugin_factory.h"

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData,
                                               size_t serialLength)
{
    assert(isPlugin(layerName));
    if (std::string(layerName).find("leaky") != std::string::npos)
    {
        assert(m_LeakyReLUCount >= 0 && m_LeakyReLUCount <= m_MaxLeakyLayers);
        assert(m_LeakyReLULayer[m_LeakyReLUCount] == nullptr);
        m_LeakyReLULayer[m_LeakyReLUCount]
            = unique_ptr_INvPlugin(nvinfer1::plugin::createPReLUPlugin(serialData, serialLength));
        ++m_LeakyReLUCount;
        return m_LeakyReLULayer[m_LeakyReLUCount - 1].get();
    }
    else if (std::string(layerName).find("reorg") != std::string::npos)
    {
        assert(m_ReorgLayer == nullptr);
        m_ReorgLayer = unique_ptr_INvPlugin(
            nvinfer1::plugin::createYOLOReorgPlugin(serialData, serialLength));
        return m_ReorgLayer.get();
    }
    else if (std::string(layerName).find("region") != std::string::npos)
    {
        assert(m_RegionLayer == nullptr);
        m_RegionLayer = unique_ptr_INvPlugin(
            nvinfer1::plugin::createYOLORegionPlugin(serialData, serialLength));
        return m_RegionLayer.get();
    }
    else
    {
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char* name)
{
    return ((std::string(name).find("leaky") != std::string::npos)
            || (std::string(name).find("reorg") != std::string::npos)
            || (std::string(name).find("region") != std::string::npos));
}

void PluginFactory::destroyPlugin()
{
    m_ReorgLayer.release();
    m_ReorgLayer = nullptr;
    m_RegionLayer.release();
    m_RegionLayer = nullptr;

    for (int i = 0; i < m_MaxLeakyLayers; ++i)
    {
        m_LeakyReLULayer[i].release();
        m_LeakyReLULayer[i] = nullptr;
    }

    m_LeakyReLUCount = 0;
}
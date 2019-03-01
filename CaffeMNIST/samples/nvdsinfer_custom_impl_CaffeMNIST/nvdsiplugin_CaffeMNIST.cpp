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

#include "factoryCaffeMNISTLegacy.h"
#include "nvdsinfer_custom_impl.h"

// Uncomment to use the legacy IPluginFactory interface
#define USE_LEGACY_IPLUGIN_FACTORY

bool NvDsInferPluginFactoryCaffeGet(NvDsInferPluginFactoryCaffe& pluginFactory,
                                    NvDsInferPluginFactoryType& type)
{
#ifdef USE_LEGACY_IPLUGIN_FACTORY
    type = PLUGIN_FACTORY;
    pluginFactory.pluginFactory = new CaffeMNISTPluginFactoryLegacy;
#else
    type = PLUGIN_FACTORY_V2;
    assert(0);
#endif

    return true;
}

void NvDsInferPluginFactoryCaffeDestroy(NvDsInferPluginFactoryCaffe& pluginFactory)
{
#ifdef USE_LEGACY_IPLUGIN_FACTORY
    CaffeMNISTPluginFactoryLegacy* factory
        = static_cast<CaffeMNISTPluginFactoryLegacy*>(pluginFactory.pluginFactory);
#else
    assert(0);
#endif
    factory->destroyPlugin();
    delete factory;
}

#ifdef USE_LEGACY_IPLUGIN_FACTORY
bool NvDsInferPluginFactoryRuntimeGet(nvinfer1::IPluginFactory*& pluginFactory)
{
    pluginFactory = new CaffeMNISTPluginFactoryLegacy;
    return true;
}

void NvDsInferPluginFactoryRuntimeDestroy(nvinfer1::IPluginFactory* pluginFactory)
{
    CaffeMNISTPluginFactoryLegacy* factory
        = static_cast<CaffeMNISTPluginFactoryLegacy*>(pluginFactory);
    factory->destroyPlugin();
    delete factory;
}
#endif

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "nvdssample_BodyPose3DNet_common.h"

#include <cassert>
#include <cstring>
#include <memory>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

class FRCNNPluginFactory : public nvcaffeparser1::IPluginFactoryV2
{
public:
  virtual nvinfer1::IPluginV2* createPlugin(const char* layerName,
      const nvinfer1::Weights* weights, int nbWeights,
      const char* libNamespace) noexcept override
  {
    assert(isPluginV2(layerName));
    if (!strcmp(layerName, "RPROIFused"))
    {
      assert(mPluginRPROI == nullptr);
      assert(nbWeights == 0 && weights == nullptr);

      auto creator = getPluginRegistry()->getPluginCreator("RPROI_TRT", "1");

      nvinfer1::PluginField fields[]{
          {"poolingH", &poolingH,  nvinfer1::PluginFieldType::kINT32, 1},
          {"poolingW", &poolingW,  nvinfer1::PluginFieldType::kINT32, 1},
          {"featureStride", &featureStride,  nvinfer1::PluginFieldType::kINT32, 1},
          {"preNmsTop", &preNmsTop,  nvinfer1::PluginFieldType::kINT32, 1},
          {"nmsMaxOut", &nmsMaxOut,  nvinfer1::PluginFieldType::kINT32, 1},
          {"iouThreshold", &iouThreshold,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"minBoxSize", &minBoxSize,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"spatialScale", &spatialScale,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"anchorsRatioCount", &anchorsRatioCount,  nvinfer1::PluginFieldType::kINT32, 1},
          {"anchorsRatios", anchorsRatios,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"anchorsScaleCount", &anchorsScaleCount,  nvinfer1::PluginFieldType::kINT32, 1},
          {"anchorsScales", anchorsScales,  nvinfer1::PluginFieldType::kFLOAT32, 1},
      };
      nvinfer1::PluginFieldCollection pluginData;
      pluginData.nbFields = 12;
      pluginData.fields = fields;

      mPluginRPROI = std::unique_ptr<IPluginV2, decltype(pluginDeleter)>(
              creator->createPlugin(layerName, &pluginData),
              pluginDeleter);
      mPluginRPROI.get()->setPluginNamespace(libNamespace);
      return mPluginRPROI.get();
    }
    else
    {
      assert(0);
      return nullptr;
    }
  }

  // caffe parser plugin implementation
  bool isPluginV2(const char* name) noexcept override { return !strcmp(name, "RPROIFused"); }

  void destroyPlugin()
  {
    mPluginRPROI.reset();
  }

  void (*pluginDeleter)(IPluginV2*){[](IPluginV2* ptr) { ptr->destroy(); }};
  std::unique_ptr<IPluginV2, decltype(pluginDeleter)> mPluginRPROI{nullptr, pluginDeleter};

  virtual ~FRCNNPluginFactory()
  {
  }
};

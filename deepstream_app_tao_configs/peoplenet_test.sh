#!/bin/sh
################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

#This script is to download peoplenet pretrained model and run peoplenet DeepStream application

echo "==================================================================="
echo "begin download models for peopleNet "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peopleNet/V2.1
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.1.1/files/resnet34_peoplenet_pruned_int8.etlt \
  -O ../../models/tao_pretrained_models/peopleNet/V2.1/resnet34_peoplenet_pruned_int8.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.1.1/files/resnet34_peoplenet_pruned_int8_gpu.txt \
  -O ../../models/tao_pretrained_models/peopleNet/V2.1/resnet34_peoplenet_pruned_int8_gpu.txt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.1.1/files/labels.txt \
  -O ../../models/tao_pretrained_models/peopleNet/V2.1/labels.txt

echo "==================================================================="
echo "Run deepstream-app"
echo "==================================================================="
deepstream-app -c deepstream_app_source1_peoplenet.txt

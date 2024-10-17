#!/bin/sh
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

echo "==================================================================="
echo "begin downloading PeopleNet model "
echo "==================================================================="
mkdir -p ./models/peoplenet
cd ./models/peoplenet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O resnet34_peoplenet_int8.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.txt' -O resnet34_peoplenet_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=labels.txt' -O labels.txt

echo "==================================================================="
echo "begin downloading BodyPose3DNet model "
echo "==================================================================="
cd -
mkdir -p ./models/bodypose3dnet
cd ./models/bodypose3dnet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/bodypose3dnet/deployable_accuracy_onnx_1.0/files?redirect=true&path=bodypose3dnet_accuracy.onnx' -O bodypose3dnet_accuracy.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/bodypose3dnet/deployable_performance_onnx_v1.0/files?redirect=true&path=bodypose3dnet_performance.onnx' -O bodypose3dnet_performance.onnx


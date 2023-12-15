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

# Check following part for how to download the TAO 3.0 models:
# https://docs.nvidia.com/tao/tao-toolkit/text/deepstream_tao_integration.html


echo "==================================================================="
echo "begin download models for Faster-RCNN / YoloV3 / YoloV4 /SSD / DSSD / RetinaNet/ UNET/"
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models
wget https://nvidia.box.com/shared/static/hzrhk33vijf31w9nxb9c93gctu1w0spd -O models.zip
unzip -o models.zip -d ../../models/tao_pretrained_models
mv ../../models/tao_pretrained_models/models/* ../../models/tao_pretrained_models/
rm -r ../../models/tao_pretrained_models/models
rm models.zip

echo "==================================================================="
echo "begin download models for dashcamnet / vehiclemakenet / vehicletypenet"
echo " / trafficcamnet / facedetectir / facenet"
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/dashcamnet && \
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.2/zip \
-O dashcamnet_pruned_v1.0.2.zip && unzip -o dashcamnet_pruned_v1.0.2.zip -d ../../models/tao_pretrained_models/dashcamnet \
&& rm dashcamnet_pruned_v1.0.2.zip
mkdir -p ../../models/tao_pretrained_models/vehiclemakenet && \
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehiclemakenet/versions/pruned_v1.0.1/zip \
-O vehiclemakenet_pruned_v1.0.1.zip && \
unzip -o vehiclemakenet_pruned_v1.0.1.zip -d ../../models/tao_pretrained_models/vehiclemakenet && \
rm vehiclemakenet_pruned_v1.0.1.zip
mkdir -p ../../models/tao_pretrained_models/vehicletypenet && \
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehicletypenet/versions/pruned_v1.0.1/zip \
-O vehicletypenet_pruned_v1.0.1.zip && \
unzip -o vehicletypenet_pruned_v1.0.1.zip -d ../../models/tao_pretrained_models/vehicletypenet && \
rm vehicletypenet_pruned_v1.0.1.zip
mkdir -p ../../models/tao_pretrained_models/trafficcamnet && \
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.2/zip \
-O trafficcamnet_pruned_v1.0.2.zip && \
unzip -o trafficcamnet_pruned_v1.0.2.zip -d ../../models/tao_pretrained_models/trafficcamnet && \
rm trafficcamnet_pruned_v1.0.2.zip
mkdir -p ../../models/tao_pretrained_models/facedetectir && \
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facedetectir/versions/pruned_v1.0.1/zip \
-O facedetectir_pruned_v1.0.1.zip && \
unzip -o facedetectir_pruned_v1.0.1.zip -d ../../models/tao_pretrained_models/facedetectir && \
rm facedetectir_pruned_v1.0.1.zip
mkdir -p ../../models/tao_pretrained_models/facenet && \
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/zip \
-O facenet_pruned_quantized_v2.0.1.zip && \
unzip facenet_pruned_quantized_v2.0.1.zip -d ../../models/tao_pretrained_models/facenet && \
rm facenet_pruned_quantized_v2.0.1.zip

echo "==================================================================="
echo "begin download models for peopleNet "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peopleNet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.3/zip \
-O peoplenet_pruned_quantized_decrypted_v2.3.3.zip && \
unzip -o peoplenet_pruned_quantized_decrypted_v2.3.3.zip -d ../../models/tao_pretrained_models/peopleNet/ && \
rm peoplenet_pruned_quantized_decrypted_v2.3.3.zip

echo "==================================================================="
echo "begin download models for peopleSegNet "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peopleSegNet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.2/zip \
-O peoplesegnet_deployable_v2.0.2.zip && \
unzip -o peoplesegnet_deployable_v2.0.2.zip -d ../../models/tao_pretrained_models/peopleSegNet/ && \
rm peoplesegnet_deployable_v2.0.2.zip

echo "==================================================================="
echo "Download models successfully "
echo "==================================================================="

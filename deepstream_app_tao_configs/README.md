################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

This document describes the procedure to download and run the Transfer Learning
Toolkit pre-trained purpose-built models in DeepStream.

The following pre-trained models are provided:
- DashCamNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:dashcamnet)
- VehicleMakeNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:vehiclemakenet)
- VehicleTypeNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:vehicletypenet)
- TrafficeCamNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet)
- PeopleNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplenet)
- FaceDetectIR (https://ngc.nvidia.com/catalog/models/nvidia:tao:facedetectir)
- PeopleSegNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesegnet)
- PeopleSemSegnet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet)

*******************************************************************************************
Downloading the config files
*******************************************************************************************
Config files are now present in https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps
under `deepstream_app_tao_configs` folder. There are two ways to obtain these configs:
1. Clone `deepstream_reference_apps` repo as per instructions in the README.
   Run the following commands:
   $ cd /opt/nvidia/deepstream/deepstream/
   $ sudo cp \
     sources/apps/sample_apps/deepstream_reference_apps/* \
     samples/configs/tao-pretrained-models/`
2. Run the following commands:
   $ cd /opt/nvidia/deepstream/deepstream/samples/configs/
   $ sudo apt-get install git-svn
   $ git svn clone \
     https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/trunk/deepstream_app_tao_configs
   $ mv deepstream_app_tao_configs tao-pretrained-models

NOTE: The above commands will require `sudo` or root permissions

*******************************************************************************
Downloading the models
*******************************************************************************
The models can be downloaded by running the following commands from the same
directory as this README. This will ensure that the models are downloaded to
the paths that the config files expect:

apt install wget
apt install zip
./download_models.sh

For how to download TAO3.0 models,
please refer https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/release/tao3.0#2-download-models to get the models

*******************************************************************************
Running the models in DeepStream:
*******************************************************************************
- An nvinfer configuration file (config_infer_*) is provided for each of the
  models.
- Following deepstream-app configuration files are provided:
  - deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt
  - deepstream_app_source1_peoplenet.txt
  - deepstream_app_source1_facedetectir.txt
  - deepstream_app_source1_trafficcamnet.txt
- For detction models, use following deepstream-app configs, default model is
  Faster Rcnn change the config-path under primary-gie group to switch to
  other models(ssd/dssd/retinanet/yolov3/yolov4/detectnet_v2/frcnn)
  - deepstream_app_source1_detection_models.txt
- Make sure encoded TAO model file paths and the INT8 calibration file paths
  are correct in the config_infer_* files and that the files exist.
- To re-use the engine files built in the first run, make sure the
  model-engine-file paths are correct in config_infer_* and deepstream_app_*
  configuration files.
- For classifier model(multi-task), use deepstream_app_source1_classifier.txt
- For instance segmentation models(MaskRCNN/peopelSegNet), use deepstream_app_source1_mrcnn.txt
  It also requires TRT plugin using https://github.com/NVIDIA/TensorRT.
  Follow
   - https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/TRT-OSS/Jetson for Jetson
   - https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/TRT-OSS/x86 for x86
- Run deepstream-app using one of the deepstream_app_* configuration files.
  $ deepstream-app -c <deepstream_app_config>
  e.g.
  $ deepstream-app -c deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt

  NOTE: Sample images/clips for FaceDetectIR would be available on it's NGC
        page.

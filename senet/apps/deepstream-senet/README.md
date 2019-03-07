# Squeeze-and-Excitation Networks as Secondary Inference Using DeepStream #

## Getting Started ##

This reference application is to demonstrate the usage of various Deepstream SDK elements in the video stream and analytics pipeline.

This app is composed of two stages:
1. Primary Object Detection (pgie)

   For the simplicity, we use Resnet10 model for the object detection since Deepstream has trained model as sample.
   You can find related files under `path/to/Deeepstream/samples/models/Primary_Detector/resnet10`

   This primary detector performs as 4 class detector (Vehicle , RoadSign, TwoWheeler,
   Person).

2. Secondary Classification for the detected objects. (sgie)

   For the secondary classifier, we use Squeeze-and-Excitation Networks trained on ImageNet.

In this deepstream reference app, we use multiple instances of "nvinfer" element. Every
instance is configured through its respective configuration file.
We provide sample configuration files under `senet/configs`
 - config_infer_primary_resnet10.txt for the primary detector
 - config_infer_secondary_senet.txt for the secondary classifier

- Note that you can use different networks for both primary and secondary gie if you have trained models.

## Installing Pre-requisites: ##

In order to use SeNet-Deepstream Reference app,

- Install Deepstream 3.0
- Install Cuda 10.0
- Install TensorRT 5.x
- Install Opencv 4.x
- Install GStreamer pre-requisites using:   
   `$ sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

- Install Google flags using:   
   `$ sudo apt-get install libgflags-dev`

- Run trt-senet app   
  We need to have SeNet TensorRT Engine file to run this application.
  Thus, you need to follow the instruction in [trt-senet README](../trt-senet/README.md)
  Make sure you have .engine file and check the path where it is stored.

## Update configurations ##
1. Modify the provided configuration files under `senet/configs`
  - config_infer_primary_resnet10.txt
    `model-file, proto-file, labelfile-path, int8-calib-file` need to be modified.
    For resnet10 example, all files are located in `path/to/Deeepstream/samples/models/Primary_Detector/resnet10`

  - config_infer_secondary_senet.txt for the secondary classifier
    `model-engine-file` needs to be modified.
     You should enter the path to TensorRT engine file we got by running trt-senet app.

2. You can change some configurations in `deepstream-senet-app.cpp` as needed.
   This cpp file contains configuration variables such as path to configuration file for pgie, label files for pgie and sgie.

## Building and running the deepstream-senet-app ##
  Run the following command to build/install the deepstream-senet-app using cmake and execute the app.
    ```
    $ cd apps/deepstream-senet
    $ mkdir build && cd build
    $ cmake -D DS_SDK_ROOT=/opt/DeepStream_Release -D CMAKE_BUILD_TYPE=Release ..
    $ make
    $ make && sudo make install
    $ cd ../../../  
    $ deepstream-senet-app <Platform> <h264_elementary_stream> <path_to_secondary_classifier_configuration_file>
    ```

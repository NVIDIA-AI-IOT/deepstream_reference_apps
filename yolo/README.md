
# Yolo Reference Application using TensorRT 5 and DeepStream SDK 3.0 #

## Setup ##

Install pre-requisites using: `$sh prebuild.sh`


Refer to the table below for information regarding installation dependencies. To use just the stand alone trt-yolo-app, Deepstream Installation can be skipped.  See [Note](https://github.com/vat-nvidia/deepstream-plugins#note) for additional installation caveats.

Platform | Applications | Deepstream Version | Dependancies | Repo Release Tag |
:-------:|:------------:|:------------------:|:------------:|:----------------:|
dGPU     | All apps and plugins | DS 3.0   |  DS 3.0, Cuda 10, TensorRT 5, OpenCV 3.3 | TOT
dGPU     | All apps and plugins | DS 2.0   |  DS 2.0, Cuda 9.2, TensorRT 4, OpenCV 3.3 | [DS2 Release](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2)
dGPU     | trt-yolo-app  | Not required | Cuda 10, TensorRT 5, OpenCV 3.3 | TOT
jetson-TX1/TX2    | nvgstiva-app, yoloplugin and trt-yolo-app | DS 1.5 | DS 1.5, Jetpack 3.3 (Cuda 9.0, TensorRT 4, OpenCV 3.3) | [DS2 Release](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2)
jetson-TX1/TX2    | trt-yolo-app | Not required | Jetpack 3.3 (Cuda 9.0, TensorRT 4, OpenCV 3.3) | [DS2 Release](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2)
jetson-Xavier   |  yoloplugin  |   DS 3.0    | Jetpack 4.1 (Cuda 10.0, TensorRT 5, OpenCV 3.3) | TOT
jetson-Xavier   | trt-yolo-app |   DS 3.0    | Jetpack 4.1 (Cuda 10.0, TensorRT 5, OpenCV 3.3) | TOT


### Building NvYolo Plugin ###

NvYolo is an inference plugin similar nvinfer in the DeepStream SDK. We make use of cmake to build all the plugins and apps in this project.

For dGPU's,   
    Set the DS_SDK_ROOT variable to point to your DeepStream SDK Root. There is also an option of using custom build paths for TensorRT(-D TRT_SDK_ROOT)and OpenCV(-D OPENCV_ROOT). These are optional and not required if the libraries have already been installed.

   `$ cd plugins/gst-yoloplugin-tesla`   
   `$ mkdir build && cd build`   
   `$ cmake -D DS_SDK_ROOT=<DS SDK Root> -D CMAKE_BUILD_TYPE=Release ..`   
   `$ make && sudo make install`

For jetson,   
    Set the DS_SDK_ROOT variable to point to your DeepStream SDK Root.

   `$ cd plugins/gst-yoloplugin-tegra`   
   `$ mkdir build && cd build`   
   `$ cmake -D DS_SDK_ROOT=<DS SDK Root> -D CMAKE_BUILD_TYPE=Release ..`   
   `$ make && sudo make install`

## Object Detection using DeepStream ##

![sample output](.sample_screen.png)

There are multiple apps that can be used to perform object detection in deepstream.

### deepstream-yolo-app ###

The deepStream-yolo-app located at `apps/deepstream_yolo` is a sample app similar to the Test-1 & Test-2 apps available in the DeepStream SDK. Using the yolo app we build a sample gstreamer pipeline using various components like H264 parser, Decoder, Video Converter, OSD and Yolo plugin to run inference on an elementary h264 video stream.

`$ cd apps/deepstream-yolo`   
`$ mkdir build && cd build`   
`$ cmake -D DS_SDK_ROOT=<DS SDK Root> -D CMAKE_BUILD_TYPE=Release ..`   
`$ make && sudo make install`   
`$ cd ../../../`   
`$ deepstream-yolo-app <Platform-Telsa/Tegra> <H264 filename> <yolo-plugin config file>`

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.

### trt-yolo-app ###

The trt-yolo-app located at `apps/trt-yolo` is a sample standalone app, which can be used to run inference on test images. This app does not have any deepstream dependencies and can be built independently. There is also an option of using custom build paths for TensorRT(-D TRT_SDK_ROOT)and OpenCV(-D OPENCV_ROOT). These are optional and not required if the libraries have already been installed.

`$ cd apps/trt-yolo`    
`$ mkdir build && cd build`   
`$ cmake -D CMAKE_BUILD_TYPE=Release ..`
`$ make && sudo make install`   
`$ cd ../../../`   
`$ trt-yolo-app --flagfile=/path/to/config-file.txt`

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.    
Test images for inference are to be added in the `test_images.txt` file in `data/`directory. Additionally run `$ trt-yolo-app --help` for a complete list of config parameters.

#### Note ####

1. If you want to generate your own calibration table, use the `calibration_images.txt` file to list of images to be used for calibration and delete the default calibration table.

2. If you want to use the `nvyolo` plugin with the deepstream-app, you will need to modify the deepstream-apps' source code to read nvyolo plugins' properties and add it to the pipeline. You can refer the `ds-example` plugin in `deepstream_config_file_parser.c` and make equivalent changes required for the `nvyolo` plugin. Specifically refer to `parse_dsexample` function and its usage in `deepstream_app_config_parser.c`

2. Tegra users working with Deepstream 1.5 and Jetpack 3.3 will have to regenerate the `.cache` files to use the standard caffe models available in the SDK. This can be done by deleting all the `.cache` files in `/home/nvidia/Model` directory and all its subdirectories and then running the nvgstiva-app using the default config file.

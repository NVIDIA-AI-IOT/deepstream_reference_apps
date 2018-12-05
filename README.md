
# Reference Apps for Video Analytics using TensorRT 5 and DeepStream SDK 3.0 #

![DS3 Workflow](.DS3-workflow.png)

## Installing Pre-requisites: ##

Install GStreamer pre-requisites using:   
   `$ sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

Install Google flags using:   
   `$ sudo apt-get install libgflags-dev`

To use just the stand alone trt-yolo-app, Deepstream Installation can be skipped. Refer to the table below for information regarding installation dependencies. See [Note](https://github.com/vat-nvidia/deepstream-plugins#note) for additional installation caveats.

Platform | Applications | Deepstream Version | Dependancies | Repo Release Tag |
:-------:|:------------:|:------------------:|:------------:|:----------------:|
dGPU     | All apps and plugins | DS 3.0   |  DS 3.0, Cuda 10, TensorRT 5 | TOT    |
dGPU     | All apps and plugins | DS 2.0   |  DS 2.0, Cuda 9.2, TensorRT 4 | [DS2 Release](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2)
dGPU     | trt-yolo-app  | Not required | Cuda 10, TensorRT 5 | TOT |
tegra    | nvgstiva-app, yoloplugin and trt-yolo-app | DS 1.5 | DS 1.5, Jetpack 3.3 (Cuda 9.0, TensorRT 4) | [DS2 Release](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2)
tegra    | trt-yolo-app | Not required | Jetpack 3.3 (Cuda 9.0, TensorRT 4) | [DS2 Release](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2)


Tegra users working with Deepstream 1.5 and Jetpack 3.3 will have to regenerate the `.cache` files to use the standard caffe models available in the SDK. This can be done by deleting all the `.cache` files in `/home/nvidia/Model` directory and all its subdirectories and then running the nvgstiva-app using the default config file.

## Setup ##

Update all the parameters in `Makefile.config` file present in the root directory

### Building NvYolo Plugin ###

1. Go to the `data/yolo` directory and add your yolo .cfg and .weights file.

    For yolo v2,   
    `$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg`   
    `$ wget https://pjreddie.com/media/files/yolov2.weights`

    For yolo v2 tiny,   
    `$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg`   
    `$ wget https://pjreddie.com/media/files/yolov2-tiny.weights`

    For yolo v3,    
    `$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg`   
    `$ wget https://pjreddie.com/media/files/yolov3.weights`

    For yolo v3 tiny,   
    `$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg`   
    `$ wget https://pjreddie.com/media/files/yolov3-tiny.weights`

2. Add absolute paths of images to be used for calibration in the `calibration_images.txt` file within the `data/yolo` directory.

3. For dGPU's,   
   `$cd sources/plugins/gst-yoloplugin-tesla`   
   `make && sudo make install`

4. For tegra,   
   `$cd sources/plugins/gst-yoloplugin-tegra`   
   `make && sudo make install`

## Object Detection using DeepStream ##

![sample output](.sample_screen.png)

There are multiple apps that can be used to perform object detection in deepstream.

### deepstream-yolo-app [TESLA platform only] ###

The deepStream-yolo-app located at `sources/apps/deepstream_yolo` is a sample app similar to the Test-1 & Test-2 apps available in the DeepStream SDK. Using the yolo app we build a sample gstreamer pipeline using various components like H264 parser, Decoder, Video Converter, OSD and Yolo plugin to run inference on an elementary h264 video stream.

`$ cd sources/apps/deepstream-yolo`   
`$ make && sudo make install`   
`$ cd ../../../`   
`$ deepstream-yolo-app /path/to/sample_video.h264 /path/to/yolo-plugin-config-file.txt`

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.

### nvgstiva-app [TEGRA platform only] ###

1.  The section below in the config file corresponds to ds-example(yolo) plugin in deepstream.
    The config file is located at `config/nvgstiva-app_yolo_config.txt`. Make any changes
    to this section if required.

    ```
    [ds-example]
    enable=1
    processing-width=640
    processing-height=480
    full-frame=1
    unique-id=15
    ```

2.  Update path to the video file source in the URI field under `[source0]` group
    of the config file
    `uri=file://path/to/source/video`

3.  Go to the root folder of this repo and run
    `$ nvgstiva-app -c config/nvgstiva-app_yolo_config.txt`

## Object Detection using trt-yolo-app ##

The trt-yolo-app located at `sources/apps/trt-yolo` is a sample standalone app, which can be used to run inference on test images. This app does not have any deepstream dependencies and can be built independently.

`$ cd sources/apps/trt-yolo`    
`$ make && sudo make install`   
`$ cd ../../../`   
`$ trt-yolo-app --flagfile=/path/to/config-file.txt`

Refer to sample config files `yolov2.txt`, `yolov2-tiny.txt`, `yolov3.txt` and `yolov3-tiny.txt` in `config/` directory.

### Note ###

1. If you want to use the `nvyolo` plugin with the deepstream-app, you will need to modify the deepstream-apps' source code to read nvyolo plugins' properties and add it to the pipeline. You can refer the `ds-example` plugin in `deepstream_config_file_parser.c` and make equivalent changes required for the `nvyolo` plugin. Specifically refer to `parse_dsexample` function and its usage in `deepstream_app_config_parser.c`
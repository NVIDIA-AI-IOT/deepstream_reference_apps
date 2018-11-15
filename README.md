
# Reference Apps for Video Analytics using TensorRT 5 and DeepStream SDK 3.0 #

![DS3 Workflow](.DS3-workflow.png)

## Installing Pre-requisites: ##

If the target platform is a dGPU, download and install DeepStream 3.0. For Tegra platforms, flash your device with Jetpack 3.3 and install Deepstream 1.5.

Install GStreamer pre-requisites using:   
   `$ sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

Install Google flags using:   
   `$ sudo apt-get install libgflags-dev`

To use just the stand alone trt-yolo-app, Deepstream Installation can be skipped. However CUDA 10.0 and TensorRT 5 should be installed. See [Note](https://github.com/vat-nvidia/deepstream-plugins#note) for additional installation caveats.

## Setup ##

Update all the parameters in `Makefile.config` file present in the root directory

### Building NvYolo Plugin ###

1. Go to the `data` directory and add your yolo .cfg and .weights file.

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

4. Set the right macro in the `network_config.h` file to choose a model architecture

5. [OPTIONAL] Update the paths of the .cfg and .weights file and other network params in `network_config.cpp` file if required.

6. Add absolute paths of images to be used for calibration in the `calibration_images.txt` file within the `data` directory.

7. Run the following command from `sources/plugins/gst-yoloplugin-tesla` for dGPU's or from `sources/plugins/gst-yoloplugin-tegra` for tegra devices to build and install the plugin
    `make && sudo make install` 

## Object Detection using DeepStream ##

![sample output](.sample_screen.png)

There are multiple apps that can be used to perform object detection in deepstream.

### deepstream-yolo-app ###

The deepStream-yolo-app located at `sources/apps/deepstream_yolo` is a sample app similar to the Test-1 & Test-2 apps available in the DeepStream SDK. Using the yolo app we build a sample gstreamer pipeline using various components like H264 parser, Decoder, Video Converter, OSD and Yolo plugin to run inference on an elementary h264 video stream.

`$ cd sources/apps/deepstream-yolo`   
`$ make && sudo make install`   
`$ cd ../../../`   
`$ deepstream-yolo-app /path/to/sample_video.h264`

### deepstream-app [TESLA platform only] ###

Following steps describe how to run the YOLO plugin in the deepstream-app

1.  The section below in the config file corresponds to ds-example(yolo) plugin in deepstream.
    The config file is located at `config/deepstream-app_yolo_config.txt`. Make any changes
    to this section if required.

    ```
    [ds-example]
    enable=1
    processing-width=1280
    processing-height=720
    full-frame=1
    unique-id=15
    gpu-id=0
    ```

2.  Update path to the video file source in the URI field under `[source0]` group
    of the config file
    `uri=file://relative/path/to/source/video`

3.  Go to the root folder of this repo and run
    `$ deepstream-app -c config/deepstream-app_yolo_config.txt`

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

The trt-yolo-app located at `sources/apps/trt-yolo` is a sample standalone app, which can be used to run inference on test images. This app does not have any deepstream dependencies and can be built independently. Add a list of absolute paths of images to be used for inference in the `test_images.txt` file located at `data` and run `trt-yolo-app` from the root directory of this repo. Additionally, the detections on test images can be saved by setting `kSAVE_DETECTIONS` config param to `true` in `network_config.cpp` file. The images overlayed with detections will be saved in the `data/detections/` directory.

This app has three command line arguments(optional). 

    1. batch_size - Integer value to be used for batch size of TRT inference. Default value is 1.   
    2. decode - Boolean value representing if the detections have to be decoded. Default value is true.   
    3. seed - Integer value to set the seed of random number generators. Default value is `time(0)`.

`$ cd sources/apps/trt-yolo`    
`$ make && sudo make install`   
`$ cd ../../../`

To run the app with default arguments   
`$ trt-yolo-app`

To change the batch_size of the TRT engine   
`$ trt-yolo-app --batch_size=4`


### Note ###

1. If you are using the plugin with deepstream-app (located at `/usr/bin/deepstream-app`), register the yolo plugin as dsexample. To do so, replace line 671 in `gstyoloplugin.cpp` with `return gst_element_register(plugin, "dsexample", GST_RANK_PRIMARY, GST_TYPE_YOLOPLUGIN);`

This registers the plugin with the name `dsexample` so that the deepstream-app can pick it up and add to it's pipeline. Now go to `sources/gst-yoloplugin/` and run `$ make && sudo make install` to build and install the plugin.

2. Tegra users working with Deepstream 1.5 and Jetpack 3.3 will have to regenerate the `.cache` files to use the standard caffe models available in the SDK. This can be done by deleting all the `.cache` files in `/home/nvidia/Model` directory and all its subdirectories and then running the nvgstiva-app using the default config file.

3. Tesla users working with Deepstream 2.0/TensorRT 4.x/CUDA 9.2, checkout the [DS2 version](https://github.com/vat-nvidia/deepstream-plugins/releases/tag/DS2) of this repo to avoid any build conflicts.
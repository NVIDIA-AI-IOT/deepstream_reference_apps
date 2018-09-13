
# YOLO Plugin for DeepStream SDK #

## Installing Pre-requisites: ##

Download and install DeepStream 2.0

Install GStreamer pre-requisites using:     
   `sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

Install Google flags using:
   `sudo apt-get install libgflags-dev`

## Setup ##

In the `Makefile.config` file present in the root directory update install paths for all the dependencies    

### Building NvYolo Plugin ###

1. Go to the `sources/gst-yoloplugin/yoloplugin_lib/data` directory and add your yolo .cfg and .weights file.
   
    For yolo v2,   
    Download the config file from the darknet repo located at `https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg`     
    Download the weights file by running the command `wget https://pjreddie.com/media/files/yolov2.weights`    

    For yolo v3,    
    Download the config file from the darknet repo located at `https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg`     
    Download the weights file by running the command `wget https://pjreddie.com/media/files/yolov3.weights`    

4. Set the right macro in the `network_config.h` file to choose a model architecture

5. [OPTIONAL] Update the paths of the .cfg and .weights file and other network params in `network_config.cpp` file if required.

7. Add absolute paths of images to be used for calibration in the `calibration_images.txt` file within the `sources/gst-yoloplugin/yoloplugin_lib/data` directory.

8. Run the following command from `sources/gst-yoloplugin/yoloplugin_lib` to build and install the plugin   
    `make && sudo make install` 

## Building and running the TRT Yolo App ##

Go to the `sources/apps/TRT-yolo` directory   

Run the following command to build and install the TRT-yolo-app   
    `make && sudo make install`

The TRT Yolo App located at `sources/apps/TRT-yolo` is a sample standalone app, which can be used to run inference on test images. This app does not have any deepstream dependencies and can be built independently. Add a list of absolute paths of images to be used for inference in the `test_images.txt` file located at `yoloplugin_lib/data/` and run `TRT-yolo-app` from the root directory of this repo. Additionally, the detections on test images can be saved by setting `kSAVE_DETECTIONS` config param to `true` in `network_config.cpp` file. The images overlayed with detections will be saved in the `yoloplugin_lib/detections/` directory.    

This app has three command line arguments(optional) that you can pass. One is the batch_size to be used for the TRT engine which is set to 1 by default. The second one is a boolean argument representing if the detections have to be decoded or not which is set to true by default. The last one is an argument to set the seed of random number generators used in the application. It is set to `time(0)` by default. To run the app with the default options run the following command from the root directory of this repo   
    `TRT-yolo-app`

To change the batch_size of the TRT engine use the following command   
    `TRT-yolo-app --batch_size=4`

## Building and Running DeepStream Yolo App ##

Go to the `sources/apps/deepstream-yolo` directory   

Run the following command to build and install the deepstream-yolo-app   
    `make && sudo make install`

The DeepStream Yolo App located at `sources/apps/deepstream_yolo` is a sample app similar to the Test-1 & Test-2 apps available in the DeepStream SDK. Using the yolo app we build a sample gstreamer pipeline using various components like H264 parser, Decoder, Video Converter, OSD and Yolo plugin to run inference on an elementary h264 video stream.

Once you have built the deepstream-yolo-app as described above, go to the root directory of this repo and run the command   
`deepstream-yolo-app /path/to/sample_video.h264`   

## Running the DeepStream App ##

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
    `deepstream-app -c config/deepstream-app_yolo_config.txt`

### Note ###

1. If you are using the plugin with deepstream-app (located at `/usr/bin/deepstream-app`), register the yolo plugin as dsexample. To do so, replace line       671 in `gstyoloplugin.cpp` with `return gst_element_register(plugin, "dsexample", GST_RANK_PRIMARY, GST_TYPE_YOLOPLUGIN);`

    This registers the plugin with the name `dsexample` so that the deepstream-app can pick it up and add to it's pipeline. Now go to `sources/gst-yoloplugin/` and run `make && sudo make install` to build and install the plugin.

2. Tegra users who are currently using Deepstream 1.5, please use the standalone TRT app as your starting point and incorporate that inference pipeline in     your inference plugin.
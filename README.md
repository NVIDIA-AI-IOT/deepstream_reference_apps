
# YOLO Plugin for DeepStream SDK #

## Installing Pre-requisites: ##

Download and install DeepStream 2.0

Install GStreamer pre-requisites using:     
   `sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

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

### Building DeepStream Yolo App ###

Go to `sources/apps/deepstream_yolo` and run the following command to build and install the deepstream-yolo-app   
    `make && sudo make install`

## Running the DeepStream Yolo App ##

The DeepStream Yolo App is a sample app similar to the Test-1 & Test-2 apps available in the DeepStream SDK. Using the yolo app we build a sample gstreamer pipeline using various components like H264 parser, Decoder, Video Converter, OSD and Yolo plugin to run inference on an elementary h264 video stream.

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

If you are using the plugin with deepstream-app (located at `/usr/bin/deepstream-app`), register the yolo plugin as dsexample. To do so, replace line 671 in `gstyoloplugin.cpp` with   
`return gst_element_register(plugin, "dsexample", GST_RANK_PRIMARY, GST_TYPE_YOLOPLUGIN);`

This registers the plugin with the name `dsexample` so that the deepstream-app can pick it up and add to it's pipeline. Now go to `sources/gst-yoloplugin/` and run the following command to build and install the plugin   
`make && sudo make install`

# YOLO Plugin for DeepStream SDK #

### Pre-requisites: ###

- DeepStream 2.0

### Installing Pre-requisites: ###

Download and install DeepStream 2.0

Install pre-requisites using:     
   `sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

Update path to TensorRT and DeepStream install directories in makefiles of both
gst-yoloplugin and the yoloplugin_lib
  ```
  TENSORT_INSTALL_DIR:= /path/to/TensorRT-4.0/
  DEEPSTREAM_INTALL_DIR:= /path/to/DeepStream_Release_2.0/
  ```
Compiling and installing the plugin:     
`Run make and sudo make install`

### Running the DeepStream App ###

Following steps describe the changes required to run YOLO in the deepstream app

1.  The section below in the config file corresponds to ds-example plugin in deepstream.
    The config file is located at config/deepstream-app_yolo_config.txt. Make any changes
    to this section if required.

    ```
    [ds-example]
    enable=1
    processing-width=640
    processing-height=480
    full-frame=1
    unique-id=15
    gpu-id=0
    ```

2.  Update path to the video file source in the URI field under [source0] group
    of the config file
    `uri=file://relative/path/to/source/video`

3.  Go to the sources/gst-yoloplugin/yoloplugin_lib/data folder and add your yolo .cfg and .weights file.

    For yolo v2 run the following commands from within the /data folder to get the config and weight files   
    `wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg`     
    `wget https://pjreddie.com/media/files/yolov2.weights`

    For yolo v3 run the following commands from within the /data folder to get the config and weight files    
    `wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg`     
    `wget https://pjreddie.com/media/files/yolov3.weights`

4.  Set the right macro in the network_config.h file to choose a model architecture

5.  Update the paths of the .cfg and weights file and other network params in network_config.cpp file(if needed).

6.  Add absolute paths of images to be used for calibration in the calibrationImages.txt file within the data folder.

7.  Go the sources/gst-yoloplugin directory and run    
    `make && sudo make install`

8.  Go to the root folder of this repo and run     
    `deepstream-app -c config/deepstream-app_yolo_config.txt`
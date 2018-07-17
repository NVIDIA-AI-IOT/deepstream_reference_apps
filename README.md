
# YOLO Plugin for DeepStream SDK #

### Pre-requisites: ###

- DeepStream 2.0

### Installing Pre-requisites: ###

Download and install DeepStream 2.0

Install GStreamer pre-requisites using:     
   `sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev`

Update path to TensorRT and DeepStream install directories in makefiles of both
gst-yoloplugin and the yoloplugin_lib
  ```
  TENSORT_INSTALL_DIR:= /path/to/TensorRT-4.0/
  DEEPSTREAM_INTALL_DIR:= /path/to/DeepStream_Release_2.0/
  ```

### Running the DeepStream App ###

Following steps describe how to run the YOLO plugin in the deepstream app

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

3.  Go to the `sources/gst-yoloplugin/yoloplugin_lib/data` folder and add your yolo .cfg and .weights file.

    For yolo v2,   
    Download the config file from the darknet repo located at `https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg`     
    Download the weights file by running the command `wget https://pjreddie.com/media/files/yolov2.weights`    

    For yolo v3,    
    Download the config file from the darknet repo located at `https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg`     
    Download the weights file by running the command `wget https://pjreddie.com/media/files/yolov3.weights`    

4.  Set the right macro in the `network_config.h` file to choose a model architecture

5.  Update the paths of the .cfg and weights file and other network params in `network_config.cpp` file if required.

7.  Add absolute paths of images to be used for calibration in the `calibrationImages.txt` file within the `sources/gst-yoloplugin/yoloplugin_lib/data` folder.

8.  Go the `sources/gst-yoloplugin` directory and run    
    `make && sudo make install`

9.  Go to the root folder of this repo and run     
    `deepstream-app -c config/deepstream-app_yolo_config.txt`
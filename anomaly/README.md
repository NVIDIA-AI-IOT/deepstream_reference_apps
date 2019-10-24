# ANOMALY DETECTION REFERENCE APP USING DEEPSTREAMSDK 4.0

## Introduction
The project contains anomaly detection application and auxiliary plug-ins to show the
capability of Deepstream SDK.

## Prequisites:
DeepStream SDK installed which is available at  http://developer.nvidia.com/deepstream-sdk
Please follow instructions in the `apps/sample_apps/deepstream-app/README` on how
to install the prequisites for Deepstream SDK apps.

## Getting Started

- Export the environment variable:
  `export DS_SDK_ROOT="your deepstream SDK root"`

- Preferably clone the app in
  `$DS_SDK_ROOT/sources/apps/sample_apps/`

- Edit the `dsanomaly_pgie_config.txt` according to the location of the models to be used


## Compilation Steps for dsdirection plugin
```
  $ cd plugins/gst-dsdirection/
  $ make && sudo make install
```

1. Test direction calculation on one video input, on dGPU, run following commands
```
gst-launch-1.0 filesrc location = samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
nvstreammux name=m batch-size=1 width=1920 height=1080 ! nvinfer config-file-path= samples/configs/deepstream-app/config_infer_primary.txt  \
! nvof ! tee name=t ! queue ! nvofvisual ! nvmultistreamtiler width=1920 height=1080 !  nveglglessink t. ! queue ! dsdirection ! \
nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! nveglglessink
```
2. Test direction calculation on one video input, on Jetson, run following commands
```
gst-launch-1.0 filesrc location = samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
nvstreammux name=m batch-size=1 width=1280 height=720 ! nvinfer config-file-path= samples/configs/deepstream-app/config_infer_primary.txt  \
! nvof ! tee name=t ! queue ! nvofvisual ! nvmultistreamtiler width=1920 height=1080 !  nvegltransform ! nveglglessink t. ! queue ! dsdirection ! \
nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! nvegltransform ! nveglglessink
```

3. Test direction calculation using optical flow on two video inputs on dGPU, run following commands
```
gst-launch-1.0 filesrc location = samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
nvstreammux name=m batch-size=2 width=1920 height=1080 ! nvinfer config-file-path= samples/configs/deepstream-app/config_infer_primary.txt ! \
nvof ! tee name=t ! queue ! nvofvisual ! nvmultistreamtiler width=1920 height=540 !  nveglglessink t. ! queue ! dsdirection ! \
nvmultistreamtiler width=1920 height=540 ! nvvideoconvert ! nvdsosd ! nveglglessink filesrc location = samples/streams/sample_1080p_h264.mp4 ! \
qtdemux ! h264parse ! nvv4l2decoder ! m.sink_1  --gst-debug=3

```
Anomaly detection app pipeline:
![DS Anomaly Detection Pipeline](.dsdirection_pipeline.png)

## Compilation Steps for Application:
```
 $ cd apps/deepstream-anomaly-detection-test/
 $ make
 $ deepstream-anomaly-detection-app <uri1> [uri2] ... [uriN]
```
  The result should be like below:
  ![DS Anomaly Detection Screenshot](.opticalflow.png)

## NOTE:
- Minimum supported resolution: DGPU - 160 x 64, Jetson - 256 x 96
- Due to an issue in nvofvisual plugin, when using nvofvisual along with nvof
  plugin, the width of input to nvof should be multiple of 32 on DGPU and multiple
  of 256 on Jetson. This will be fixed in the nvofvisual plugin in the next DeepStream
  release.

# RUNTIME SOURCE ADDITION DELETION REFERENCE APP USING DEEPSTREAMSDK 6.4

## Introduction
The project contains Runtime source addition/deletion application to show the
capability of Deepstream SDK.

## Prerequisites:
DeepStream SDK installed which is available at  http://developer.nvidia.com/deepstream-sdk
Please follow instructions in the apps/sample_apps/deepstream-app/README on how
to install the prequisites for Deepstream SDK apps.

## Getting Started

- Preferably clone the app in
  `/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/`

- Edit all the inference models config files according to the location of the models to be used

## Compilation Steps and Execution:
```
  $ Set CUDA_VER in the MakeFile as per platform.
      For both x86 & Jetson, CUDA_VER=12.2
  $ sudo make

  $ ./deepstream-test-rt-src-add-del <uri> <run forever> <sink> <sync>
  $ ./deepstream-test-rt-src-add-del file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4 0 nveglglessink 1 #dGPU - nveglglessink Jetson - nv3dsink
  $ ./deepstream-test-rt-src-add-del rtsp://127.0.0.1/video 0 nveglglessink 1 #dGPU
```

The application demonstrates following pipeline for single source <uri>

uridecodebin -> nvstreammux -> nvinfer -> nvtracker -> nvtiler -> nvvideoconvert -> nvdsosd -> displaysink

- At runtime after a timeout a source will be added periodically. All the components
  are reconfigured during addition/deletion
- After reaching of `MAX_NUM_SOURCES`, each source is deleted periodically till single
  source is present in the pipeline
- The app exits, when final source End of Stream is reached or if the last source is deleted.
- filesink and nv3dsink (only Jetson) are also supported.






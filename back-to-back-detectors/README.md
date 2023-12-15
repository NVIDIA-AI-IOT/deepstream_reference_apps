# BACK-TO-BACK-DETECTORS REFERENCE APP USING DEEPSTREAMSDK 6.4

## Introduction
The project contains Back to Back detector application to show the
capability of Deepstream SDK.

This sample builds on top of the deepstream-test1 sample to demonstrate how to
add multiple back-to-back detectors in the pipeline.

Two instances of "nvinfer" or "nvinferserver" element are added to the pipeline serially after
nvstreammux and before the display components. Both the "nvinfer" or "nvinferserver" instances have
their own config files.

The first "nvinfer" or "nvinferserver" instance (Person/Vehicle/Bicycle/RoadSign) will always act
as primary detector.

The second "nvinfer" or "nvinferserver" instance (Face Detection) can be configured as
primary(full-frame) / secondary (operating on primary detected objects). By
default it is configured in the secondary mode. To change the second "nvinfer" or "nvinferserver"
instance to primary mode, change the macro `SECOND_DETECTOR_IS_SECONDARY` in the
sources to 0.

## Prequisites:

Please follow instructions in the `apps/sample_apps/deepstream-app/README` on how
to install the prequisites for Deepstream SDK, the DeepStream SDK itself and the
apps.

## Getting Started

- Preferably clone the app in
  `/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/`

- Edit the `primary_detector_config.txt` according to the location of the models to be used

## Steps to download the models:
- To download the models for the second nvinfer, visit:
  https://github.com/NVIDIA-AI-IOT/redaction_with_deepstream
-  Use the following commands:
```
  $ cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/
  $ cd deepstream_reference_apps/deepstream_app_tao_configs/
  $ sudo cp -a * /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
  $ sudo apt install -y wget zip
  $ cd /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
  $ sudo ./download_models.sh

Below steps are applicable if you want to run inference using nvinferserver inside DeepStream Triton based docker:

- Setup Triton model repository:
  $ cd /opt/nvidia/deepstream/deepstream/samples/
- Run prepare_ds_triton_model_repo.sh script to create Primary infer model "PrimaryDetector"
  $ ./prepare_ds_triton_model_repo.sh
- Run prepare_ds_triton_tao_model_repo.sh script to create Secondary infer model "FaceNet"
  $ ./prepare_ds_triton_tao_model_repo.sh

```

Back to back detectors app pipeline:
![DS Back to back detectors Pipeline](.backtobackdetectors_pipeline.png)

The result should be like below:
![DS Back to back detectors Screenshot](.backtobackdetectors.png)
## Compilation Steps and Execution:
```
  $ Set CUDA_VER in the MakeFile as per platform.
      For both x86 & Jetson, CUDA_VER=12.2
  $ sudo make

  $ ./back-to-back-detectors <h264_elementary_stream>
    Ex.: ./back-to-back-detectors /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264

Use option "-t inferserver" to select nvinferserver as the inference plugin
  $ ./back-to-back-detectors -t inferserver <h264_elementary_stream>
    Ex.: ./back-to-back-detectors -t inferserver /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264

NOTE:
- For Jetson, first run the below commands
  $ cd /opt/nvidia/deepstream/deepstream/samples/triton_tao_model_repo
  $ sudo ln -s ../triton_model_repo/Primary_Detector .
Then run the app.

```
The result should be like below:
  ![DS Back to Back Detectors Screenshot](.backtobackdetectors.png)

NOTE:
- Run the above commands with sudo.
- Edit the paths in `secondary_detector_config.txt` to the location of the models
  downloaded from the above site.
- back-to-back-detectors application does not run inside jetson triton docker.

This document shall describe about the sample back-to-back-detectors application.

This sample builds on top of the deepstream-test1 sample to demonstrate how to
add multiple back-to-back detectors in the pipeline.

Two instances of the "nvinfer" element are added to the pipeline serially after
nvstreammux and before the display components. Both the "nvinfer" instances have
their own config files.

The first "nvinfer" instance (Person/Vehicle/Bicycle/RoadSign) will always act
as primary detector.

The second "nvinfer" instance (Face/License Plate) can be configured as
primary(full-frame) / secondary (operating on primary detected objects). By
default it is configured in the secondary mode. To change the second "nvinfer"
instance to primary mode, change the macro `SECOND_DETECTOR_IS_SECONDARY` in the
sources to 0.


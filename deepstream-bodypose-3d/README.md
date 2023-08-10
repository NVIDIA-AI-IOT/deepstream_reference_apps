# 3d-bodypose-deepstream

## Introduction
The project contains 3D Body Pose application built using  Deepstream SDK.

This application is built for [KAMA: 3D Keypoint Aware Body Mesh Articulation](https://arxiv.org/abs/2104.13502).
![sample pose output](./sources/.screenshot.png)
## Prerequisites:
DeepStream SDK 6.3 installed which is available at  http://developer.nvidia.com/deepstream-sdk
Please follow instructions in the `/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/README` on how
to install the prequisites for building Deepstream SDK apps.

## Installation
Follow https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html to setup the DeepStream SDK

1. Preferably clone the app in
  `/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/`
and define project home as `export BODYPOSE3D_HOME=<parent-path>/deepstream-bodypose-3d`.

2. Install [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) and download [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) and [BodyPose3DNet](https://ngc.nvidia.com/models/nvstaging:tao:bodypose3dnet) from NGC.
```bash
$ mkdir -p $BODYPOSE3D_HOME/models
$ cd $BODYPOSE3D_HOME/models
# Download PeopleNet
$ ngc registry model download-version "nvidia/tao/peoplenet:deployable_quantized_v2.5"
# Download BodyPose3DNet
$ ngc registry model download-version "nvidia/tao/bodypose3dnet:deployable_accuracy_v1.0"
```

By now the directory tree should look like this
```bash
$ tree $BODYPOSE3D_HOME -d
$BODYPOSE3D_HOME
├── configs
├── models
│   ├── bodypose3dnet_vdeployable_accuracy_v1.0
│   └── peoplenet_vdeployable_quantized_v2.5
├── sources
│   ├── deepstream-sdk
│   └── nvdsinfer_custom_impl_BodyPose3DNet
└── streams
```

3. Download and extract [Eigen 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page) under the project foler.
```bash
$ cd $BODYPOSE3D_HOME
$ wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
$ tar xvzf eigen-3.4.0.tar.gz
$ ln eigen-3.4.0 eigen -s
```

4. For Deepstream SDK version older than 6.2, copy and build custom `NvDsEventMsgMeta` into Deepstream SDK installation path. Copy and build custom `NvDsEventMsgMeta` into Deepstream SDK installation path.
The custom `NvDsEventMsgMeta` structure handles pose3d and pose25d meta data.
```bash
# Copy deepstream sources
cp $BODYPOSE3D_HOME/sources/deepstream-sdk/eventmsg_payload.cpp /opt/nvidia/deepstream/deepstream/sources/libs/nvmsgconv/deepstream_schema
# Build new nvmsgconv library for custom Product metadata
cd /opt/nvidia/deepstream/deepstream/sources/libs/nvmsgconv
make; make install
```
Please note that this step is not necessary for Deepstream SDK version 6.2 or newer.

## Build the applications
```bash
# Build custom nvinfer parser of BodyPose3DNet
cd $BODYPOSE3D_HOME/sources/nvdsinfer_custom_impl_BodyPose3DNet
make
# Build deepstream-pose-estimation-app
cd $BODYPOSE3D_HOME/sources
make
```
If the above steps are successful, `deepstream-pose-estimation-app` shall be built in the same directory. Under `$BODYPOSE3D_HOME/sources/nvdsinfer_custom_impl_BodyPose3DNet`, `libnvdsinfer_custom_impl_BodyPose3DNet.so` should be present as well.

## Run the applications
### `deepstream-pose-estimation-app`
The command line options of this application are listed below:
```bash
$ ./deepstream-pose-estimation-app -h
Usage:
  deepstream-pose-estimation-app [OPTION?] Deepstream BodyPose3DNet App

Help Options:
  -h, --help                        Show help options
  --help-all                        Show all help options
  --help-gst                        Show GStreamer Options

Application Options:
  -v, --version                     Print DeepStreamSDK version.
  --version-all                     Print DeepStreamSDK and dependencies version.
  --input                           [Required] Input video address in URI format by starting with "rtsp://" or "file://".
  --output                          Output video address. Either "rtsp://" or a file path is acceptable. If the value is "rtsp://", then the result video is published at "rtsp://localhost:8554/ds-test".
  --save-pose                       The file path to save both the pose25d and the recovered pose3d in JSON format.
  --conn-str                        Connection string for Gst-nvmsgbroker, e.g. <ip address>;<port>;<topic>.
  --publish-pose                    Specify the type of pose to publish. Acceptable value is either "pose3d" or "pose25d". If not specified, both "pose3d" and "pose25d" are published to the message broker.
  --tracker                         Specify the NvDCF tracker mode. The acceptable value is either "accuracy" or "perf". The default value is "accuracy".
  --fps                             Print FPS in the format of current_fps (averaged_fps).
  --width                           Input video width in pixels. The default value is 1280.
  --height                          Input video height in pixels. The default value is 720.
  --focal                           Camera focal length in millimeters. The default value is 800.79041.
```

Here are examples running this application:
1. Below command processes an input video in URI format and renders the overlaid pose estimation in a window.
```bash
$ ./deepstream-pose-estimation-app --input file://$BODYPOSE3D_HOME/streams/bodypose.mp4
```
Please provide the absolute path to the source video file.

2. When the data source is a video file, below command saves the output video with the skeleton overlay to `$BODYPOSE3D_HOME/streams/bodypose_3dbp.mp4` and save the skeleton's keypoints to `$BODYPOSE3D_HOME/streams/bodypose_3dbp.json`.
```bash
$ ./deepstream-pose-estimation-app --input file://$BODYPOSE3D_HOME/streams/bodypose.mp4 --output $BODYPOSE3D_HOME/streams/bodypose_3dbp.mp4 --focal 800.0 --width 1280 --height 720 --fps --save-pose $BODYPOSE3D_HOME/streams/bodypose_3dbp.json
```
`bodypose_3dbp.json` contains the predicted 34 keypoints in both `pose25d` and `pose3d` space:
```bash
[{
  "num_frames_in_batch": 1,
  "batches": [{
    "batch_id": 0,
    "frame_num": 3,
    "ntp_timestamp": 1639431716322229000,
    "num_obj_meta": 6,
    "objects": [{
      "object_id": 3,
      "pose25d": [707.645203, 338.592499, -0.000448, 0.867188, ...],
      "pose3d": [297.649933, -94.196518, 3520.129883, 0.867188, ...]
    },{
    ...
    }]
  }]
}, {
```
`pose25d` contains `34x4` floats. A four-item group represents a keypoint's `[x, y, zRel, conf]`
values. `x` and `y` are the keypoint's position in the image coordinate; `zRel` is the relative
depth value from the skeleton's root keypoint, i.e. pelvis. `x, y, zRel` values are in millimeters.
`conf` is the confidence value of the prediction.

`pose3d` also contains `34x4` floats. A four-item group represents a keypoint's `[x, y, z, conf]`
values. `x`, `y`, `z` are the keypoint's 3D position in the world coordinate whose origin is the
camera. `x, y, z` values are in millimeters. `conf` is the confidence value of the prediction.

3. When the data source is an RTSP stream and the result is published to RTSP stream `rtsp://localhost:8554/ds-test`,
```bash
$ ./deepstream-pose-estimation-app --input rtsp://<ipa_address>:<port>/<topic> --output rtsp://
```

4. In order to publish both pose3D and pose25D metadata to a message broker, please do
```bash
$ ./deepstream-pose-estimation-app --input file://$BODYPOSE3D_HOME/streams/bodypose.mp4 --conn-str "localhost;9092;test"
```
where `\"localhost;9092;test\"` is the connection string to the message broker `localhost`, port number `9092`, and topic name `test`. Please apply double quotes around the connection string since `;` is a reserved character in shell.

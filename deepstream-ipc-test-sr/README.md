#  IPC Test Supper Resolution
## Introduction
This sample demonstrates how to Zero-copy share decoded buffers over IPC and how to integarte Super-Resolution 
model. This sample can support Jetson and DGPU platform.
The client pipeline looks like "......-> nvstreammux -> pgie ->nvvideoconvert -> capsfilter -> nvvideotemplate + ......".

## Prerequisites

Please follow instructions in the /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/README on how
to install the prerequisites for the Deepstream SDK, the DeepStream SDK itself, and the apps.

You must have the following development packages installed
   GStreamer-1.0
   GStreamer-1.0 Base Plugins
   GStreamer-1.0 gstrtspserver
   X11 client-side library

To install these packages, execute the following command:
   sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev

## Build

```bash
  $ export CUDA_VER=xx.x # For DS8.0 on x86 CUDA_VER=12.8, on Jetson CUDA_VER=13.0
  $ sudo make            # (sudo not required in case of docker containers)
```

NOTE: To compile the sources, run make with "sudo" or root permission.
      To improve performance on specific GPUs, please add "-gencode=arch=compute_xx,code=sm_xx" in Makefile. Computing capability can be found in this link https://developer.nvidia.com/zh-cn/cuda-gpus#compute.

## Generate super resolution model

The model is from pytorch [code](https://github.com/pytorch/tutorials/blob/2f4e5c368a2754276d70cf4aed4a97bcf01ed551/advanced_source/super_resolution_with_onnxruntime.py). 
Here are the steps to generate the model.

```bash
  $ pip install onnx onnxruntime torch torchvision
  $ git clone --shallow-since=2025-07-10  https://github.com/pytorch/tutorials.git
  $ cd tutorials && git reset --hard 876c56359626a9716e524ec832674f26272ad13a
  $ python3 advanced_source/super_resolution_with_onnxruntime.py
  # After generating super_resolution.onnx in the current directory,
  # copy the model to deepstream-ipc-test-sr directory.
  $ cp advanced_source/super_resolution.onnx /path/to/your/deepstream-ipc-test-sr
```

## Run

Run with the command line. This sample act as either server or client based on command line arguments.

```shell
    # server
    $ ./deepstream-ipc-test-app server <url> <domain_socket_path>
    # client
    $ ./deepstream-ipc-test-app client <domain_socket_path>
```
e.g.

- Server generates a url using a local file. Multiple clients play the url.

```shell
    $ ./deepstream-ipc-test-app server file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /tmp/test1
    $ ./deepstream-ipc-test-app client /tmp/test1
    $ ./deepstream-ipc-test-app client /tmp/test1
```

- Server generates a url using RTSP. Client plays the url.

```shell
    $ ./deepstream-ipc-test-app server rtsp://127.0.0.1/video1 /tmp/test1
    $ ./deepstream-ipc-test-app client /tmp/test1
```

- Server generates two urls. Client plays these two urls.

```shell
    $ ./deepstream-ipc-test-app server rtsp://127.0.0.1/video1 /tmp/test1 rtsp://127.0.0.1/video2 /tmp/test2
    $ ./deepstream-ipc-test-app client /tmp/test1 /tmp/test2
```

The server accepts H.264/H.265 video stream RTSP URL and IPC socket path
as input. It does the decoding of the stream and listens for the connection
on the IPC socket path. It sends decoded data over IPC to the connected client.

The client accepts IPC socket path as input. It sends connection request to the
server. Once server accepts the request, it starts receiving the decoded data
over IPC which is further pushed to deepstream pipeline. The rest of the pipeline
is similar to the deepstream-test3 sample.

## Performance
### FPS Measurement
The client supports FPS statistic.

```shell
$ IPC_SR_PERF_MODE=1 ./deepstream-ipc-test-app server \
    file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /tmp/test1 \
    file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /tmp/test2 \
    file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /tmp/test3 \
    file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /tmp/test4

$ IPC_SR_PERF_MODE=1 ./deepstream-ipc-test-app client \
                                     /tmp/test1 /tmp/test2 \
                                     /tmp/test3 /tmp/test4 | grep FPS
```

The client will output FPS in the terminal log, such as
```
AVG FPS (num_sources 4 * 65.370): 261.481
AVG FPS (num_sources 4 * 65.631): 262.525
AVG FPS (num_sources 4 * 65.758): 263.031
AVG FPS (num_sources 4 * 65.880): 263.518
```

FPS statistic of the client:
|Device            |  FPS(batch_size:4)  |
| ---------------- | ----- |
|A40               |  263.780  |
|Thor              |   306.331  |

### Latency Measurement
On the server side, Add `GstReferenceTimestampMeta` in the probe function of `nvunixfdsink` sink pad and serialize it through `serialize_meta`.
On the client side, Deserialize through `deserialize_meta`, and get the sending time in the probe function of the src pad of `nvunixfdsrc`

Start sever with the following command.

```shell
$ IPC_SR_PERF_MODE=1 ./deepstream-ipc-test-app server file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 /tmp/test1
```
Open another terminal, start client with the following command.

```shell
$ IPC_SR_PERF_MODE=1 ./deepstream-ipc-test-app client /tmp/test1 | grep latency
```

The client will output latency in the terminal log, such as

```
IPC latency 0.406
IPC latency 0.423
IPC latency 0.429
IPC latency 0.467
IPC latency 0.402
IPC latency 0.403
IPC latency 0.402
IPC latency 0.385
```
The latency of IPC is related to both CPU and GPU.

|Device            |  Latency  |
| ---------------- | --------- |
|A40 & AMD 7232P   |  ~0.4ms   |
|Thor              |  ~0.117ms |

NOTE:
- On Thor, it is a known issue that `Latency Measurement` is unavailable.
- To reuse engine files generated in previous runs, update the
model-engine-file parameter in the nvinfer config file to an existing
engine file.
- The engine model should be present to run IPC use-case. If it is not
present, the IPC test will timeout in first run as it takes some time
to generate the model.
- This example only support nvinfer.

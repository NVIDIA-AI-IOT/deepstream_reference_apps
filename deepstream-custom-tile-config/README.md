# DeepStream Custom Tiler Configuration Sample

## Introduction
This sample demonstrates the usage of "custom-tile-config" of nvmultistreamtiler to customize the tiling positions and sizes of multiple videos within the display window. The rectangle display areas of every video can be configured by the "custom-tile-config" property of nvmultistreamtiler for CustomTileConfig struct. 

## Prerequisites
The sample works with DeepStream 9.0 GA or above version. Please follow [DeepStream SDK installation instruction](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) or use [DeepStream docker container](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html) to prepare the DeepStream environment.

## Running the Application
Download the source code and build

``
git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git
cd deepstream_reference_apps/deepstream-custom-tile-config
make
``

Run the sample with four video files and generate mp4 video for the output

``
./deepstream-custom-tile-config -i file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 --no-display
``
Run the sample with four video files and dislay on the screen
``
./deepstream-custom-tile-config -i file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4
``

## Known Issue
The video blending is not supported now. If there are overlapping parts of the videos, the overlapping parts will flicker. 
This sample shows how to use custom TensorRT PLAN files within the NvInfer plugin to perform inference on video streams. We take an example of YoloV3 network and use this network to perform object detection in deepstream. Along with the plugin factory implementation required for the network, the sample also provides the output bounding box parser functions which parse the output buffers of the network and generate its corresponding meta data.

Pre-requisites:
- Generate a TensorRT plan file for YoloV3 using the `trt-yolo-app` present at `sources/apps/trt-yolo-app` and update its corresponding path for `model-engine-file` property in `deepstream_app_config_yoloV3.txt` and `config_infer_primary_YoloV3.txt` files. 
- Update the `uri` field in `[source0]` element to point towards a input video source file.

We use cmake to compile the library:   
Set the DS_SDK_ROOT variable to point to your DeepStream SDK Root. There is also an option of using custom build paths for TensorRT(-D TRT_SDK_ROOT). This is optional and not required if the libraries have already been installed.

   `$ cd nvdsinfer_custom_impl_YoloV3`
   `$ mkdir build && cd build`   
   `$ cmake -D DS_SDK_ROOT=<DS SDK Root> -D CMAKE_BUILD_TYPE=Release ..`   
   `$ make`
   `$ cd ../../`

Run the sample:   
   `$ deepstream-app -c deepstream_app_config_yoloV3.txt`
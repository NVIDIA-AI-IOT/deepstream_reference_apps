This sample shows how to use custom TensorRT PLAN files within the NvInfer plugin to perform inference on video streams. We take an example of YoloV3 network and use this network to perform object detection in deepstream. Along with the plugin factory implementation required for the network, the sample also provides the output bounding box parser functions which parse the output buffers of the network and generate its corresponding meta data.

Pre-requisites:
- Generate a TensorRT plan file for YoloV3 using the `trt-yolo-app` present at `sources/apps/trt-yolo-app` and update its corresponding path for `model-engine-file` property in `deepstream_app_config_yoloV3.txt` and `config_infer_primary_YoloV3.txt` files. 
- Update the `uri` field in `[source0]` element to point towards a input video source file.

Compile the library:   
$ make -C nvdsinfer_custom_impl_YoloV3

Run the sample:   
$deepstream-app -c deepstream_app_config_yoloV3.txt
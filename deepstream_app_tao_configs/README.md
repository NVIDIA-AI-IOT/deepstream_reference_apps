This document describes the procedure to download and run the TAO pre-trained purpose-built models in DeepStream.

The following pre-trained models are provided:

##### Detection Network

- Faster-RCNN / YoloV3 / SSD / DSSD / RetinaNet / YoloV4 / YoloV4-tiny
- DetectNet_v2 (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_detectnet_v2)

##### Classification Network

- multi_task (https://ngc.nvidia.com/catalog/models/nvidia:tao:pretrained_image_classification)

##### Other Networks

- DashCamNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:dashcamnet)
- VehicleMakeNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:vehiclemakenet)
- VehicleTypeNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:vehicletypenet)
- TrafficeCamNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet)
- PeopleNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplenet)
- FaceDetectIR (https://ngc.nvidia.com/catalog/models/nvidia:tao:facedetectir)
- PeopleSegNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesegnet)
- PeopleSemSegnet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet)

*******************************************************************************************
## 1. Download the config files

*******************************************************************************************
```
$ git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git
$ cd deepstream_reference_apps/deepstream_app_tao_configs/
$ sudo apt install -y wget zip
```

*******************************************************************************************
## 2. Prepare Pretrained Models
*******************************************************************************************
Choose one of the following three inferencing methods:
- For TensorRT based inferencing, please run the following commands
```
$ sudo cp -a * /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
$ cd /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
$ sudo ./download_models.sh
```
- For Triton Inference Server based inferencing, the DeepStream application works as the Triton client:
  * To set up the native Triton Inference Sever, please refer to [triton_server.md](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream_app_tao_configs/triton_server.md).
  * To set up the separated Triton Inference Sever, please refer to [triton_server_grpc.md](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream_app_tao_configs/triton_server_grpc.md)


For more information on TAO models,
please refer https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps#2-download-models.



*******************************************************************************
## 3. Run the models in DeepStream

*******************************************************************************
```
$ sudo deepstream-app -c deepstream_app_source1_$MODEL.txt
```
e.g.
```
$ sudo deepstream-app -c deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt
```
The yaml config files can also be used
```
$ sudo deepstream-app -c deepstream_app_source1_$MODEL.yml
```
e.g.
```
$ sudo deepstream-app -c deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.yml
```

**Note:**

1. Modify the corresponding parts in the *deepstream_app_source1_$MODEL.txt* configuration to choose Which model or which inferencing module will be use, please search the **[primary-gie]** section in the configuration file, for example

   First choose the inferencing method with setting "plugin-type" and then choose which model will be used. Then the corresponding "config-file" can be set. The following is the sample of nvinferserver native inferencing based Yolov4 model ineferncing configuration. 
   ```
   [primary-gie]
   enable=1
   gpu-id=0
   #(0): nvinfer; (1): nvinferserver
   plugin-type=1
   # Modify as necessary
   batch-size=1
   #Required by the app for OSD, not a plugin property
   bbox-border-color0=1;0;0;1
   bbox-border-color1=0;1;1;1
   bbox-border-color2=0;0;1;1
   bbox-border-color3=0;1;0;1
   gie-unique-id=1
   # Replace the infer primary config file when you need to
   # use other detection models
   #config-file=nvinfer/config_infer_primary_frcnn.txt
   #config-file=triton/config_infer_primary_frcnn.txt
   #config-file=triton-grpc/config_infer_primary_frcnn.txt
   ...
   ...
   #config-file=nvinfer/config_infer_primary_yolov4.txt
   config-file=triton/config_infer_primary_yolov4.txt
   #config-file=triton-grpc/config_infer_primary_yolov4.txt
   #config-file=config_infer_primary_detectnet_v2.txt
   ...
   ```

*******************************************************************************
## 4. Related Links

*******************************************************************************
deepstream-tao-app : https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps 

TAO Toolkit Guide : https://docs.nvidia.com/tao/tao-toolkit/index.html

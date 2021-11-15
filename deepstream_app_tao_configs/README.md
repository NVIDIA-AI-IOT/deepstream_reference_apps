This document describes the procedure to download and run the TAO pre-trained purpose-built models in DeepStream.

The following pre-trained models are provided:

##### Detection Network

- Faster-RCNN / YoloV3 / SSD / DSSD / RetinaNet / YoloV4  (https://ngc.nvidia.com/catalog/models/nvidia:tao:Faster-RCNN)
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
$ sudo cp -a * /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
```

*******************************************************************************
## 2. Download the models

*******************************************************************************
```
$ sudo apt install -y wget zip
$ cd /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
$ sudo ./download_models.sh
```

For more information on TAO3.0 models,
please refer https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/release/tao3.0#2-download-models.



*******************************************************************************
## 3. Run the models in DeepStream

*******************************************************************************
```
$ sudo deepstream-app -c deepstream_app_source1_$MODEL.txt
e.g.
$ sudo deepstream-app -c deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt
```

**Note:**

1. For which model of the *deepstream_app_source1_$MODEL.txt* uses, please find from the **[primary-gie]** section in it, for example

   Below is the **[primary-gie]** config of deepstream_app_source1_detection_models.txt, which indicates it uses yolov4 by default, and user can change to frcnn/ssd/dssd/retinanet/yolov3/detectnet_v2 by commenting "config-file=config_infer_primary_yolov4.txt" and uncommenting the corresponding "config-file=" .

   ```
   [primary-gie]
   enable=1
   gpu-id=0
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
   #config-file=config_infer_primary_frcnn.txt
   #config-file=config_infer_primary_ssd.txt
   #config-file=config_infer_primary_dssd.txt
   #config-file=config_infer_primary_retinanet.txt
   #config-file=config_infer_primary_yolov3.txt
   config-file=config_infer_primary_yolov4.txt
   #config-file=config_infer_primary_detectnet_v2.txt
   ```

2. When running the model with "deepstream-app", during the TensorRT engine building stage, if you see error logs like 

   `ERROR: [TRT]: UffParser: Validator error: FirstDimTile_5: Unsupported operation _BatchTilePlugin_TRT`

   it indicates the TensorRT plugin lib needs to be updated following steps in 

    - https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/TRT-OSS/Jetson for Jetson 

    - https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/TRT-OSS/x86 for x86 


*******************************************************************************
## 4. Related Links

*******************************************************************************
deepstream-tao-app : https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps 

TAO Toolkit Guide : https://docs.nvidia.com/tao/tao-toolkit/index.html 

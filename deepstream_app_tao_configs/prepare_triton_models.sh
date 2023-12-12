#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin


if [ ! ${IS_JETSON_PLATFORM} ]; then
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_8.6.3.1_x86/files?redirect=true&path=tao-converter' -O tao-converter
else
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_jp6.0_aarch64/files?redirect=true&path=tao-converter' -O tao-converter
fi
chmod 755 tao-converter

wget https://nvidia.box.com/shared/static/hzrhk33vijf31w9nxb9c93gctu1w0spd -O models.zip
unzip -o models.zip -d ./triton/
mv ./triton/models/* ./triton/
rm -r ./triton/models
rm models.zip

mkdir -p ./triton/frcnn/1
trtexec --onnx=./triton/frcnn/frcnn_kitti_resnet18.epoch24_trt8.onnx --int8 --calib=./triton/frcnn/cal_frcnn_20230707_cal.bin --saveEngine=./triton/frcnn/1/frcnn_kitti_resnet18.epoch24_trt8.onnx_b4_gpu0_int8.engine --minShapes=input_image:1x3x544x960 --optShapes=input_image:2x3x544x960 --maxShapes=input_image:4x3x544x960&
cp triton/frcnn_config.pbtxt ./triton/frcnn/config.pbtxt

mkdir -p ./triton/dashcamnet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.2/zip \
-O dashcamnet_pruned_v1.0.2.zip && unzip -o dashcamnet_pruned_v1.0.2.zip -d ./triton/dashcamnet
rm dashcamnet_pruned_v1.0.2.zip

mkdir -p ./triton/dashcamnet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/dashcamnet/dashcamnet_int8.txt -b 1 -d 3,544,960 -e ./triton/dashcamnet/1/resnet18_dashcamnet_pruned.etlt_b1_gpu0_int8.engine ./triton/dashcamnet/resnet18_dashcamnet_pruned.etlt&
cp triton/dashcamnet_config.pbtxt ./triton/dashcamnet/config.pbtxt

mkdir -p ./triton/vehiclemakenet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehiclemakenet/versions/pruned_v1.0.1/zip \
-O vehiclemakenet_pruned_v1.0.1.zip
unzip -o vehiclemakenet_pruned_v1.0.1.zip -d ./triton/vehiclemakenet/
rm vehiclemakenet_pruned_v1.0.1.zip

mkdir -p ./triton/vehiclemakenet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/vehiclemakenet/vehiclemakenet_int8.txt -b 4 -d 3,224,224 -e ./triton/vehiclemakenet/1/resnet18_vehiclemakenet_pruned.etlt_b4_gpu0_int8.engine ./triton/vehiclemakenet/resnet18_vehiclemakenet_pruned.etlt&
cp triton/vehiclemakenet_config.pbtxt ./triton/vehiclemakenet/config.pbtxt

mkdir -p ./triton/vehicletypenet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehicletypenet/versions/pruned_v1.0.1/zip \
-O vehicletypenet_pruned_v1.0.1.zip
unzip -o vehicletypenet_pruned_v1.0.1.zip -d ./triton/vehicletypenet
rm -r vehicletypenet_pruned_v1.0.1.zip

mkdir -p ./triton/vehicletypenet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/vehicletypenet/vehicletypenet_int8.txt -b 4 -d 3,224,224 -e ./triton/vehicletypenet/1/resnet18_vehicletypenet_pruned.etlt_b4_gpu0_int8.engine ./triton/vehicletypenet/resnet18_vehicletypenet_pruned.etlt&
cp triton/vehicletypenet_config.pbtxt ./triton/vehicletypenet/config.pbtxt

mkdir -p ./triton/peopleNet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.3/zip \
-O peoplenet_pruned_quantized_decrypted_v2.3.3.zip
unzip -o peoplenet_pruned_quantized_decrypted_v2.3.3.zip -d ./triton/peopleNet/
rm peoplenet_pruned_quantized_decrypted_v2.3.3.zip

mkdir -p ./triton/peopleNet/1
trtexec --onnx=./triton/peopleNet/resnet34_peoplenet_int8.onnx --int8 --calib=./triton/peopleNet/resnet34_peoplenet_int8.txt --saveEngine=./triton/peopleNet/1/resnet34_peoplenet_int8.onnx_b1_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960&
cp triton/peopleNet_config.pbtxt ./triton/peopleNet/config.pbtxt

mkdir -p ./triton/peopleSegNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.2/zip \
-O peoplesegnet_deployable_v2.0.2.zip
unzip -o peoplesegnet_deployable_v2.0.2.zip -d ./triton/peopleSegNet/
rm peoplesegnet_deployable_v2.0.2.zip

mkdir -p ./triton/peopleSegNet/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/peopleSegNet/peoplesegnet_resnet50_int8.txt -b 1 -d 3,576,960 -e ./triton/peopleSegNet/1/peoplesegnet_resnet50.etlt_b1_gpu0_int8.engine ./triton/peopleSegNet/peoplesegnet_resnet50.etlt&
cp triton/peopleSegNet_config.pbtxt ./triton/peopleSegNet/config.pbtxt

mkdir -p ./triton/detectnet_v2/1
trtexec --onnx=./triton/detectnet_v2/detectnetv2_resnet18.onnx --int8 --calib=./triton/detectnet_v2/cal.bin \
 --saveEngine=./triton/detectnet_v2/1/detectnetv2_resnet18.onnx_b1_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 \
 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960&
cp triton/detectnet_config.pbtxt ./triton/detectnet_v2/config.pbtxt

mkdir -p ./triton/trafficcamnet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.2/zip \
-O trafficcamnet_pruned_v1.0.2.zip
unzip -o trafficcamnet_pruned_v1.0.2.zip -d ./triton/trafficcamnet/
rm trafficcamnet_pruned_v1.0.2.zip

mkdir -p ./triton/trafficcamnet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/trafficcamnet/trafficcamnet_int8.txt -b 1 -d 3,544,960 -e ./triton/trafficcamnet/1/resnet18_trafficcamnet_pruned.etlt_b1_gpu0_int8.engine ./triton/trafficcamnet/resnet18_trafficcamnet_pruned.etlt&
cp triton/trafficcamnet_config.pbtxt ./triton/trafficcamnet/config.pbtxt

mkdir -p ./triton/dssd/1
trtexec --onnx=./triton/dssd/dssd_resnet18_epoch_118.onnx --int8 --calib=./triton/dssd/dssd_cal.bin \
--saveEngine=./triton/dssd/1/dssd_resnet18_epoch_118.onnx_b4_gpu0_int8.engine \
--minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&
cp triton/dssd_config.pbtxt ./triton/dssd/config.pbtxt

mkdir -p ./triton/ssd/1
trtexec --onnx=./triton/ssd/ssd_resnet18_epoch_074.onnx --int8 --calib=./triton/ssd/ssd_cal.bin \
--saveEngine=./triton/ssd/1/ssd_resnet18_epoch_074.onnx_b4_gpu0_int8.engine \
--minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&
cp triton/ssd_config.pbtxt ./triton/ssd/config.pbtxt

mkdir -p ./triton/efficientdet/1
trtexec --onnx=./triton/efficientdet/d0_avlp_544_960.onnx --int8 \
--calib=./triton/efficientdet/d0_avlp_544_960.cal \
--saveEngine=./triton/efficientdet/1/d0_avlp_544_960.onnx_b1_gpu0_int8.engine
cp triton/efficientdet_config.pbtxt ./triton/efficientdet/config.pbtxt

mkdir -p ./triton/retinanet/1
trtexec --onnx=./triton/retinanet/retinanet_resnet18_epoch_080_its.onnx --int8 --calib=./triton/retinanet/retinanet_resnet18_epoch_080_its_tao5.cal --saveEngine=./triton/retinanet/1/retinanet_resnet18_epoch_080_its.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&
cp triton/retinanet_config.pbtxt ./triton/retinanet/config.pbtxt

mkdir -p ./triton/facedetectir/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facedetectir/versions/pruned_v1.0.1/zip \
-O facedetectir_pruned_v1.0.1.zip
unzip -o facedetectir_pruned_v1.0.1.zip -d ./triton/facedetectir/
rm facedetectir_pruned_v1.0.1.zip

mkdir -p ./triton/facedetectir/1
./tao-converter -k tlt_encode -t int8 -c ./triton/facedetectir/facedetectir_int8.txt -b 1 -d 3,240,384 \
  -e ./triton/facedetectir/1/resnet18_facedetectir_pruned.etlt_b1_gpu0_int8.engine ./triton/facedetectir/resnet18_facedetectir_pruned.etlt&
cp triton/facedetectir_config.pbtxt ./triton/facedetectir/config.pbtxt

mkdir -p ./triton/yolov3/1
trtexec --onnx=./triton/yolov3/yolov3_resnet18_398.onnx --int8 --calib=./triton/yolov3/cal.bin.trt8517 --saveEngine=./triton/yolov3/1/yolov3_resnet18_398.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960 --layerPrecisions=cls/Sigmoid:fp32,cls/Sigmoid_1:fp32,box/Sigmoid_1:fp32,box/Sigmoid:fp32,cls/Reshape_reshape:fp32,box/Reshape_reshape:fp32,Transpose2:fp32,sm_reshape:fp32,encoded_sm:fp32,conv_big_object:fp32,cls/mul:fp32,box/concat_concat:fp32,box/add_1:fp32,box/mul_4:fp32,box/add:fp32,box/mul_6:fp32,box/sub_1:fp32,box/add_2:fp32,box/add_3:fp32,yolo_conv1_6:fp32,yolo_conv1_6_lrelu:fp32,yolo_conv2:fp32,Resize1:fp32,yolo_conv1_5_lrelu:fp32,encoded_bg:fp32,yolo_conv4_lrelu:fp32,yolo_conv4:fp32&
cp triton/yolov3_config.pbtxt ./triton/yolov3/config.pbtxt

mkdir -p ./triton/yolov4/1
trtexec --onnx=./triton/yolov4/yolov4_resnet18_epoch_080.onnx --int8 --calib=./triton/yolov4/cal_trt861.bin --saveEngine=./triton/yolov4/1/yolov4_resnet18_epoch_080.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&
cp triton/yolov4_config.pbtxt ./triton/yolov4/config.pbtxt

mkdir -p ./triton/yolov4-tiny/1
trtexec --onnx=./triton/yolov4-tiny/yolov4_cspdarknet_tiny_397.onnx --int8 --calib=./triton/yolov4-tiny/cal.bin.trt8517 --saveEngine=./triton/yolov4-tiny/1/yolov4_cspdarknet_tiny_397.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&
cp triton/yolov4-tiny_config.pbtxt ./triton/yolov4-tiny/config.pbtxt

#mkdir -p ./triton/multi_task/1
#./tao-converter -k nvidia_tlt -t fp16 -b 1 -d 3,80,60 -e ../triton/multi_task/1/abc.etlt_b1_gpu0_fp16.engine ./triton/multi_task/abc.etlt
#cp triton/multi_task_config.pbtxt ./triton/multi_task/config.pbtxt


#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin


if [ ! ${IS_JETSON_PLATFORM} ]; then
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v4.0.0_trt8.5.2.2_x86/files/tao-converter' -O tao-converter
else
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_aarch64/files/tao-converter' -O tao-converter
fi
chmod 755 tao-converter

wget https://nvidia.box.com/shared/static/taqr2y52go17x1ymaekmg6dh8z6d43wr -O models.zip
unzip models.zip -d ./triton/
mv ./triton/models/* ./triton/
rm -r ./triton/models
rm models.zip

mkdir -p ./triton/frcnn/1
./tao-converter ./triton/frcnn/frcnn_kitti_resnet18.epoch24_trt8.etlt -k nvidia_tlt -t int8 -c ./triton/frcnn/cal_8517.bin -b 4 -d 3,544,960 -e ./triton/frcnn/1/frcnn_kitti_resnet18.epoch24_trt8.etlt_b4_gpu0_int8.engine
cp triton/frcnn_config.pbtxt ./triton/frcnn/config.pbtxt

mkdir -p ./triton/dashcamnet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.2/zip \
-O dashcamnet_pruned_v1.0.2.zip && unzip dashcamnet_pruned_v1.0.2.zip -d ./triton/dashcamnet
rm dashcamnet_pruned_v1.0.2.zip

mkdir -p ./triton/dashcamnet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/dashcamnet/dashcamnet_int8.txt -b 1 -d 3,544,960 -e ./triton/dashcamnet/1/resnet18_dashcamnet_pruned.etlt_b1_gpu0_int8.engine ./triton/dashcamnet/resnet18_dashcamnet_pruned.etlt
cp triton/dashcamnet_config.pbtxt ./triton/dashcamnet/config.pbtxt

mkdir -p ./triton/vehiclemakenet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehiclemakenet/versions/pruned_v1.0.1/zip \
-O vehiclemakenet_pruned_v1.0.1.zip
unzip vehiclemakenet_pruned_v1.0.1.zip -d ./triton/vehiclemakenet/
rm vehiclemakenet_pruned_v1.0.1.zip

mkdir -p ./triton/vehiclemakenet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/vehiclemakenet/vehiclemakenet_int8.txt -b 4 -d 3,224,224 -e ./triton/vehiclemakenet/1/resnet18_vehiclemakenet_pruned.etlt_b4_gpu0_int8.engine ./triton/vehiclemakenet/resnet18_vehiclemakenet_pruned.etlt
cp triton/vehiclemakenet_config.pbtxt ./triton/vehiclemakenet/config.pbtxt

mkdir -p ./triton/vehicletypenet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehicletypenet/versions/pruned_v1.0.1/zip \
-O vehicletypenet_pruned_v1.0.1.zip
unzip vehicletypenet_pruned_v1.0.1.zip -d ./triton/vehicletypenet
rm -r vehicletypenet_pruned_v1.0.1.zip

mkdir -p ./triton/vehicletypenet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/vehicletypenet/vehicletypenet_int8.txt -b 4 -d 3,224,224 -e ./triton/vehicletypenet/1/resnet18_vehicletypenet_pruned.etlt_b4_gpu0_int8.engine ./triton/vehicletypenet/resnet18_vehicletypenet_pruned.etlt
cp triton/vehicletypenet_config.pbtxt ./triton/vehicletypenet/config.pbtxt

mkdir -p ./triton/peopleNet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/zip \
-O peoplenet_deployable_quantized_v2.6.1.zip
unzip peoplenet_deployable_quantized_v2.6.1.zip -d ./triton/peopleNet/
rm peoplenet_deployable_quantized_v2.6.1.zip

mkdir -p ./triton/peopleNet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/peopleNet/resnet34_peoplenet_int8.txt -b 1 -d 3,544,960 -e ./triton/peopleNet/1/resnet34_peoplenet_int8.etlt_b1_gpu0_int8.engine ./triton/peopleNet/resnet34_peoplenet_int8.etlt
cp triton/peopleNet_config.pbtxt ./triton/peopleNet/config.pbtxt

mkdir -p ./triton/peopleSegNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.2/zip \
-O peoplesegnet_deployable_v2.0.2.zip
unzip peoplesegnet_deployable_v2.0.2.zip -d ./triton/peopleSegNet/
rm peoplesegnet_deployable_v2.0.2.zip

mkdir -p ./triton/peopleSegNet/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/peopleSegNet/peoplesegnet_resnet50_int8.txt -b 1 -d 3,576,960 -e ./triton/peopleSegNet/1/peoplesegnet_resnet50.etlt_b1_gpu0_int8.engine ./triton/peopleSegNet/peoplesegnet_resnet50.etlt
cp triton/peopleSegNet_config.pbtxt ./triton/peopleSegNet/config.pbtxt

mkdir -p ./triton/detectnet_v2/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/detectnet_v2/cal.bin -b 1 -d 3,544,960 -e ./triton/detectnet_v2/1/detectnetv2_resnet18.etlt_b1_gpu0_int8.engine ./triton/detectnet_v2/detectnetv2_resnet18.etlt
cp triton/detectnet_config.pbtxt ./triton/detectnet_v2/config.pbtxt

mkdir -p ./triton/trafficcamnet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.2/zip \
-O trafficcamnet_pruned_v1.0.2.zip
unzip trafficcamnet_pruned_v1.0.2.zip -d ./triton/trafficcamnet/
rm trafficcamnet_pruned_v1.0.2.zip

mkdir -p ./triton/trafficcamnet/1
./tao-converter -k tlt_encode -t int8 -c ./triton/trafficcamnet/trafficcamnet_int8.txt -b 1 -d 3,544,960 -e ./triton/trafficcamnet/1/resnet18_trafficcamnet_pruned.etlt_b1_gpu0_int8.engine ./triton/trafficcamnet/resnet18_trafficcamnet_pruned.etlt
cp triton/trafficcamnet_config.pbtxt ./triton/trafficcamnet/config.pbtxt

mkdir -p ./triton/dssd/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/dssd/dssd_cal.bin \
  -b 1 -d 3,544,960 -e ./triton/dssd/1/dssd.etlt_b1_gpu0_int8.engine ./triton/dssd/dssd.etlt
cp triton/dssd_config.pbtxt ./triton/dssd/config.pbtxt

mkdir -p ./triton/ssd/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/ssd/ssd_cal.bin -b 1 -d 3,544,960 \
  -e ./triton/ssd/1/ssd.etlt_b1_gpu0_int8.engine \
  ./triton/ssd/ssd.etlt
cp triton/ssd_config.pbtxt ./triton/ssd/config.pbtxt

mkdir -p ./triton/efficientdet/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/efficientdet/d0_avlp_544_960.cal -p image_arrays:0,1x3x544x960,1x3x544x960,1x3x544x960 \
  -e ./triton/efficientdet/1/d0_avlp_544_960.etlt_b1_gpu0_int8.engine ./triton/efficientdet/d0_avlp_544_960.etlt
cp triton/efficientdet_config.pbtxt ./triton/efficientdet/config.pbtxt

mkdir -p ./triton/retinanet/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/retinanet/retinanet_resnet18_epoch_080_its_trt8.cal \
 -b 1 -d 3,544,960 -e ./triton/retinanet/1/retinanet_resnet18_epoch_080_its_trt8.etlt_b1_gpu0_int8.engine \
 ./triton/retinanet/retinanet_resnet18_epoch_080_its_trt8.etlt
cp triton/retinanet_config.pbtxt ./triton/retinanet/config.pbtxt

mkdir -p ./triton/facedetectir/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facedetectir/versions/pruned_v1.0.1/zip \
-O facedetectir_pruned_v1.0.1.zip
unzip facedetectir_pruned_v1.0.1.zip -d ./triton/facedetectir/
rm facedetectir_pruned_v1.0.1.zip

mkdir -p ./triton/facedetectir/1
./tao-converter -k tlt_encode -t int8 -c ./triton/facedetectir/facedetectir_int8.txt -b 1 -d 3,240,384 \
  -e ./triton/facedetectir/1/resnet18_facedetectir_pruned.etlt_b1_gpu0_int8.engine ./triton/facedetectir/resnet18_facedetectir_pruned.etlt
cp triton/facedetectir_config.pbtxt ./triton/facedetectir/config.pbtxt

mkdir -p ./triton/yolov3/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/yolov3/cal.bin.trt8517 \
  -p Input,1x3x544x960,1x3x544x960,1x3x544x960 \
  -e ./triton/yolov3/1/yolov3_resnet18_398.etlt_b1_gpu0_int8.engine \
  ./triton/yolov3/yolov3_resnet18_398.etlt
cp triton/yolov3_config.pbtxt ./triton/yolov3/config.pbtxt

mkdir -p ./triton/yolov4/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/yolov4/cal.bin.trt8517 \
  -p Input,1x3x544x960,1x3x544x960,1x3x544x960 \
  -e ./triton/yolov4/1/yolov4_resnet18_395.etlt_b1_gpu0_int8.engine \
  ./triton/yolov4/yolov4_resnet18_395.etlt
cp triton/yolov4_config.pbtxt ./triton/yolov4/config.pbtxt

mkdir -p ./triton/yolov4-tiny/1
./tao-converter -k nvidia_tlt -t int8 -c ./triton/yolov4-tiny/cal.bin.trt8517 \
  -p Input,1x3x544x960,1x3x544x960,1x3x544x960 \
  -e ./triton/yolov4-tiny/1/yolov4_cspdarknet_tiny_397.etlt_b1_gpu0_int8.engine \
  ./triton/yolov4-tiny/yolov4_cspdarknet_tiny_397.etlt
cp triton/yolov4-tiny_config.pbtxt ./triton/yolov4-tiny/config.pbtxt

#mkdir -p ./triton/multi_task/1
#./tao-converter -k nvidia_tlt -t fp16 -b 1 -d 3,80,60 -e ../triton/multi_task/1/abc.etlt_b1_gpu0_fp16.engine ./triton/multi_task/abc.etlt
#cp triton/multi_task_config.pbtxt ./triton/multi_task/config.pbtxt


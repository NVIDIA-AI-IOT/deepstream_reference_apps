#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin

trtexec --fp16 --onnx=./models/yolov4/1/yolov4_-1_3_416_416_dynamic.onnx.nms.onnx --saveEngine=./models/yolov4/1/yolov4_-1_3_416_416_dynamic.onnx_b32_gpu0.engine  --minShapes=input:1x3x416x416 --optShapes=input:16x3x416x416 --maxShapes=input:32x3x416x416 --shapes=input:16x3x416x416

mkdir -p models/trafficcamnet/1/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned.onnx' -O ./models/trafficcamnet/1/resnet18_trafficcamnet_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned_int8.txt' -O ./models/trafficcamnet/resnet18_trafficcamnet_pruned_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=labels.txt' -O ./models/trafficcamnet/labels.txt
trtexec --onnx=./models/trafficcamnet/1/resnet18_trafficcamnet_pruned.onnx --int8 --calib=./models/trafficcamnet/resnet18_trafficcamnet_pruned_int8.txt \
 --saveEngine=./models/trafficcamnet/1/resnet18_trafficcamnet_pruned.onnx_b8_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 \
 --optShapes="input_1:0":4x3x544x960 --maxShapes="input_1:0":8x3x544x960

mkdir -p models/US_LPD/1/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.3.1/files?redirect=true&path=LPDNet_usa_pruned_tao5.onnx' -O models/US_LPD/1/LPDNet_usa_pruned_tao5.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.3.1/files?redirect=true&path=usa_cal_10.1.0.bin' -O models/US_LPD/1/usa_cal_10.1.0.bin
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/usa_lpd_label.txt' -O models/US_LPD/usa_lpd_label.txt
trtexec --onnx=models/US_LPD/1/LPDNet_usa_pruned_tao5.onnx --int8 --calib=models/US_LPD/1/usa_cal_10.1.0.bin \
 --saveEngine=models/US_LPD/1//LPDNet_usa_pruned_tao5.onnx_b16_gpu0_int8.engine --minShapes="input_1:0":1x3x480x640 \
 --optShapes="input_1:0":16x3x480x640 --maxShapes="input_1:0":16x3x480x640

mkdir models/us_lprnet/1/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lprnet/deployable_onnx_v1.1/files?redirect=true&path=us_lprnet_baseline18_deployable.onnx' -O models/us_lprnet/1/us_lprnet_baseline18_deployable.onnx
trtexec --onnx=models/us_lprnet/1/us_lprnet_baseline18_deployable.onnx --fp16 \
 --saveEngine=models/us_lprnet/1/us_lprnet_baseline18_deployable.onnx_b16_gpu0_fp16.engine --minShapes="image_input":1x3x48x96 \
 --optShapes="image_input":8x3x48x96 --maxShapes="image_input":16x3x48x96

mkdir -p models/peoplenet/1/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=resnet34_peoplenet.onnx' \
  -O models/peoplenet/1/resnet34_peoplenet.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=resnet34_peoplenet_int8.txt' \
  -O models/peoplenet/1/resnet34_peoplenet_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=labels.txt' \
  -O models/peoplenet/labels.txt
trtexec --onnx=./models/peoplenet/1/resnet34_peoplenet.onnx --int8 \
 --calib=./models/peoplenet/1/resnet34_peoplenet_int8.txt --saveEngine=./models/peoplenet/1/resnet34_peoplenet.onnx_b8_gpu0_int8.engine \
 --minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":8x3x544x960 --maxShapes="input_1:0":8x3x544x960

#generate engine for vehicle related models.
echo "Building Model Secondary_CarMake..."
mkdir -p models/Secondary_CarMake/1/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
  -O models/Secondary_CarMake/1/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
  -O models/Secondary_CarMake/labels.txt
trtexec --onnx=models/Secondary_CarMake/1/resnet18_pruned.onnx --fp16  \
        --saveEngine=models/Secondary_CarMake/1/resnet18_pruned.onnx_b16_gpu0_fp16.engine --minShapes="input_1:0":1x3x224x224 \
        --optShapes="input_1:0":8x3x224x224 --maxShapes="input_1:0":16x3x224x224

echo "Building Model Secondary_VehicleTypes..."
mkdir -p models/Secondary_VehicleTypes/1/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
  -O models/Secondary_VehicleTypes/1/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
  -O models/Secondary_VehicleTypes/labels.txt
trtexec --onnx=models/Secondary_VehicleTypes/1/resnet18_pruned.onnx --fp16 \
        --saveEngine=models/Secondary_VehicleTypes/1/resnet18_pruned.onnx_b16_gpu0_fp16.engine --minShapes="input_1:0":1x3x224x224 \
        --optShapes="input_1:0":8x3x224x224 --maxShapes="input_1:0":16x3x224x224
echo "Finished generating engine files."

#!/bin/bash

set -e

if [ ! -d peoplenet ];then
  mkdir peoplenet
fi

cd peoplenet
if [ ! -e labels.txt ];then
  echo "Downloading peoplenet label.... "
  wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/labels.txt
fi

if [ ! -e resnet34_peoplenet_pruned_int8.etlt ];then
  echo "Downloading peoplenet etlt model.... "
  wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.etlt
fi

if [ ! -e resnet34_peoplenet_pruned_int8.txt ];then
  echo "Downloading peoplenet int8 .... "
  wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.txt
fi
cd -

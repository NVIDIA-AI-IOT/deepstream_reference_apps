#!/usr/bin/env bash
# /**
# MIT License

# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *
# */

echo "Installing prerequisities ... "

# Install GStreamer pre-requisites
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

# Install Google flags and cmake using:
sudo apt-get install libgflags-dev cmake

# Download yolo weights
echo "Downloading yolo weights and config files ... "
# For yolo v2,
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg --directory-prefix=data/ -q --show-progress
wget https://pjreddie.com/media/files/yolov2.weights --directory-prefix=data/ -q --show-progress

# For yolo v2 tiny,
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg --directory-prefix=data/ -q --show-progress
wget https://pjreddie.com/media/files/yolov2-tiny.weights --directory-prefix=data/ -q --show-progress

# For yolo v3,
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg --directory-prefix=data/ -q --show-progress
wget https://pjreddie.com/media/files/yolov3.weights --directory-prefix=data/ -q --show-progress

# For yolo v3 tiny,
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg --directory-prefix=data/ -q --show-progress
wget https://pjreddie.com/media/files/yolov3-tiny.weights --directory-prefix=data/ -q --show-progress

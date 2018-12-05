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


CONFIG :=../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

ifeq ($(CUDA_VER),)
$(error "CUDA_VER variable is not set in Makefile.config")
endif

ifeq ($(PLATFORM),)
$(error PLATFORM variable is not set in Makefile.config)
endif

CUCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
SRCS:= $(wildcard *.cpp)
KERNELS:= $(wildcard *.cu)
BUILD_PATH:= ./build/
OBJS= $(patsubst %, $(BUILD_PATH)%, $(SRCS:.cpp=.o))
OBJS+=$(patsubst %, $(BUILD_PATH)%, $(KERNELS:.cu=.o))  
DEPS:= $(SRCS)
DEPS+= $(KERNELS)
DEPS+= $(wildcard *.h)
TARGET:= libyoloplugin.a

ifeq ($(PLATFORM), TESLA)
	INCS:= -I"$(TENSORRT_INSTALL_DIR)/include" \
           -I"/usr/local/cuda-$(CUDA_VER)/include" \
           -I "$(OPENCV_INSTALL_DIR)/include"

endif

ifeq ($(PLATFORM), TEGRA)
	INCS:= -I"usr/include/aarch64-linux-gnu" \
	       -I"/usr/local/cuda-$(CUDA_VER)/include" \
		   -I "/usr/include"

endif

CXXFLAGS:= -O2 -std=c++11 -g -lstdc++fs -ldl -fPIC -Wall -Wunused-function -Wunused-variable -Wfatal-errors $(shell pkg-config --cflags glib-2.0)

.PHONY: all dirs clean deps

all: dirs deps
	ar rcs $(TARGET) $(OBJS)

dirs:	
	if [ ! -d "build" ]; then mkdir -p build; fi
	if [ ! -d "../../../data/yolo/detections" ]; then mkdir -p ./../../data/yolo/detections; fi

deps: $(DEPS) $(OBJS)

$(BUILD_PATH)%.o: %.cpp %.h
	$(CXX) $(INCS) -c -o $@ $(CXXFLAGS) $<

$(BUILD_PATH)%.o: %.cu
	$(CUCC) -c -o $@ -arch=compute_50 --shared -Xcompiler -fPIC  $<
clean: 
	rm -f ./build/*
	rm -f ./*.a

clean_models:
	rm -rf ../../data/yolo/*.engine

clean_detections:
	rm -rf ../../data/yolo/detections/*
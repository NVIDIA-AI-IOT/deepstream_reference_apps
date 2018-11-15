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

CONFIG :=../../../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

ifeq ($(CUDA_VER),)
$(error "CUDA_VER variable is not set in Makefile.config")
endif

ifeq ($(CUDA_VER),"10.0")
$(error "This sample is supposed to be used with DS3.0 which requires cuda 10.0")
endif

ifeq ($(PLATFORM),)
$(error PLATFORM variable is not set in Makefile.config)
endif

ifneq ($(PLATFORM), TESLA)
$(error This app is developed for TESLA platform)
endif

ifeq (' $(DEEPSTREAM_INSTALL_DIR)', ' /path/to/DeepStream_Release_3.0')
$(error DEEPSTREAM_INSTALL_DIR variable is not set in Makefile.config)
endif

CUCC:= /usr/local/cuda-$(CUDA_VER)/bin/nvcc
CC=g++

SRCS:=  $(wildcard *.cpp)
KERNELS:= $(wildcard *.cu)
BUILD_PATH:= ./build/
OBJS= $(patsubst %, $(BUILD_PATH)%, $(SRCS:.cpp=.o))
OBJS+=$(patsubst %, $(BUILD_PATH)%, $(KERNELS:.cu=.o))  

TARGET_LIB:= libnvdsinfer_custom_impl_YoloV3.so

INCS:= -I "$(TENSORRT_INSTALL_DIR)/include" \
       -I "/usr/local/cuda-$(CUDA_VER)/include" \
	   -I $(DEEPSTREAM_INSTALL_DIR)/sources/includes 

LIBS:= -L$(TENSORRT_INSTALL_DIR)/lib -lnvinfer -lnvinfer_plugin
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

CFLAGS:= -O2 -Wall -std=c++11 -shared -fPIC

all: dirs $(TARGET_LIB)

dirs:	
	if [ ! -d "build" ]; then mkdir -p build; fi

$(BUILD_PATH)%.o: %.cpp
	$(CC) -c -o $@ $(CFLAGS) $(INCS) $< 

$(BUILD_PATH)%.o: %.cu
	$(CUCC) -c -o $@ $(INCS) -arch=compute_50 --shared -Xcompiler -fPIC $<

$(TARGET_LIB) :$(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB) $(OBJS)
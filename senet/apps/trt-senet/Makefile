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

APP:= trt-senet-app
LIB:= ../../lib
ENG:=../../data/SE-ResNet50-Engines
CALIB:=$(LIB)/calibration
SRCS:=$(wildcard *.cpp)
DEPS:=$(SRCS)
SRCS+=$(wildcard $(LIB)/*.cpp)
DEPS+=$(wildcard $(LIB)/*.h)
OBJS:=$(patsubst %.cpp,%.o,$(SRCS))

CONFIG :=../../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

INCS:=	-I"$(TENSORRT_INSTALL_DIR)/samples/common"\
		    -I"$(TENSORRT_INSTALL_DIR)/include"\
		    -I"/usr/local/cuda-$(CUDA_VER)/include"\
				-I"$(OPENCV_INSTALL_DIR)/include/"\
				-isystem"$(OPENCV_INSTALL_DIR)/include/opencv4"

LIBS:= 	-L"/usr/local/cuda-$(CUDA_VER)/lib64" -lcudart -Wl,-rpath="/usr/local/cuda-$(CUDA_VER)/lib64"\
		    -L"$(OPENCV_INSTALL_DIR)/lib" -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_dnn \
				-Wl,-rpath="$(OPENCV_INSTALL_DIR)/lib" \
		    -L$(TENSORRT_INSTALL_DIR)/lib -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser	-Wl,-rpath="$(TENSORRT_INSTALL_DIR)/lib"

CXXFLAGS:= -Wall -std=c++11 -lstdc++fs -Wall -Wunused-function -Wunused-variable -Wfatal-errors	-I $(LIB)

all: dirs $(APP)

dirs:
	if [ ! -d "../../lib/calibration" ]; then mkdir -p ../../lib/calibration; fi
	if [ ! -d "../../data" ]; then mkdir -p ../data; fi
	if [ ! -d "../../data/SE-ResNet50-Engines" ]; then mkdir -p ../../data/SE-ResNet50-Engines; fi

$(APP): $(OBJS)
	@echo Building Executable...
	$(CXX) $(OBJS) $(CXXFLAGS) $(LIBS) -o $@

%.o: %.cpp
	@echo Building object file $@
	@$(CXX) $<	$(CXXFLAGS) $(INCS)	-c -o $@

clean:
	rm -f *.o
	rm -f $(APP)
	rm -f $(LIB)/*.o

clean_engines:
	rm -f $(ENG)/*.engine

clean_calibration_tables:
	rm -f $(CALIB)/*.table

clean_all:
	rm -f *.o
	rm -f $(APP)
	rm -f $(LIB)/*.o
	rm -f $(ENG)/*.engine
	rm -f $(CALIB)/*.table

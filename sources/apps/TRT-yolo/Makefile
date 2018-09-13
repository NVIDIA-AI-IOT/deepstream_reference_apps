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

APP:= TRT-yolo-app
YOLO_LIB:= ../../gst-yoloplugin/yoloplugin_lib

OBJS = $(wildcard $(YOLO_LIB)/build/*.o)
DEPS:= $(wildcard $(YOLO_LIB)/*.cpp)
DEPS:= $(wildcard $(YOLO_LIB)/*.cu)
DEPS+= $(wildcard $(YOLO_LIB)/*.h)

CONFIG :=../../../Makefile.config
ifeq ($(wildcard $(CONFIG)),)
$(error $(CONFIG) missing.)
endif
include $(CONFIG)

CXXFLAGS:= -O2 -std=c++11 -lstdc++fs -Wall -Wunused-function -Wunused-variable -I $(YOLO_LIB) -I /usr/local/cuda-$(CUDA_VER)/include

LIBS:= -L "$(TENSORRT_INSTALL_DIR)/lib" -lnvinfer -lnvinfer_plugin -Wl,-rpath="$(TENSORRT_INSTALL_DIR)/lib" \
       -L "/usr/local/cuda-$(CUDA_VER)/lib64" -lcudart -lcublas -lcurand -Wl,-rpath="/usr/local/cuda-$(CUDA_VER)/lib64" \
       -L "$(OPENCV_INSTALL_DIR)/lib" -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_dnn -Wl,-rpath="$(OPENCV_INSTALL_DIR)/lib" \
       -L "/usr/lib/x86_64-linux-gnu" -lgflags

.PHONY: all deps install clean

all: $(APP)

$(APP): deps
	@echo Building $(APP)
	$(CXX) -o $(APP) TRT-yolo-app.cpp $(OBJS) $(CXXFLAGS) $(LIBS) 

deps: $(DEPS)
	@echo "Building yoloplugin_lib"
	$(MAKE) -C $(YOLO_LIB)

install: $(APP)
	@echo Installing $(APP)
	cp -rv $(APP) /usr/bin/

clean:
	rm -rf $(APP)
	$(MAKE) -C $(YOLO_LIB) clean
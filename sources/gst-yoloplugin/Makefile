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

CUDA_VER:=9.2
CXX:= g++
SRCS:= gstyoloplugin.cpp
INCS:= $(wildcard *.h)
LIB:=libgstnvyoloplugin.so

OPENCV_INSTALL_DIR:=/path/to/opencv-3.4.0
TENSORRT_INSTALL_DIR:= /path/to/TensorRT-4.0/
DEEPSTREAM_INTALL_DIR:= /path/to/DeepStream_Release_2.0/

DEP:=yoloplugin_lib/libyoloplugin.a
DEP_FILES:=$(wildcard yoloplugin_lib/yoloplugin_lib.* )
DEP_FILES-=$(DEP)

CFLAGS := -fPIC -std=c++11 -Wunused-variable
LIBS := -shared -Wl,-no-undefined \
  -L yoloplugin_lib -lyoloplugin \
  -L$(OPENCV_INSTALL_DIR)/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_imgcodecs \
  -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnppc -lnppig -lnpps \
  -L/usr/local/deepstream/ -lgstnvquery -lgstnvdsmeta \
  -Wl,-rpath,/usr/local/deepstream/ \
  -L $(TENSORRT_INSTALL_DIR)/lib -lnvinfer -lnvinfer_plugin


CFLAGS+= \
	-I /usr/local/cuda-$(CUDA_VER)/include \
	-I $(DEEPSTREAM_INTALL_DIR)/sources/includes \
    -I $(OPENCV_INSTALL_DIR)/include \
	-I $(TENSORRT_INSTALL_DIR)/include


OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(DEP) $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

$(DEP): $(DEP_FILES)
	$(MAKE) -C yoloplugin_lib/

install: $(LIB)
	cp -rv $(LIB) /usr/lib/x86_64-linux-gnu/gstreamer-1.0/

clean:
	rm -rf $(OBJS) $(LIB)
	rm -f ./yoloplugin_lib/obj/*.o
	rm -f ./yoloplugin_lib/*.a

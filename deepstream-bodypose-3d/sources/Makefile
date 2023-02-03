################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX=g++ -std=c++14

APP:= deepstream-pose-estimation-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CUDA_HOME:= /usr/local/cuda-$(CUDA_VER)
DEEPSTREAM_HOME:= /opt/nvidia/deepstream/deepstream

LIB_INSTALL_DIR?=$(DEEPSTREAM_HOME)/lib/
APP_INSTALL_DIR?=$(DEEPSTREAM_HOME)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= deepstream_pose_estimation_app.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(patsubst %.c,%.o, $(patsubst %.cpp,%.o, $(SRCS)))

CFLAGS+= \
	-I$(CUDA_HOME)/include \
	-I$(DEEPSTREAM_HOME)/sources/includes \
	-I$(DEEPSTREAM_HOME)/sources/apps/apps-common/includes \
	-I$(DEEPSTREAM_HOME)/sources/apps/sample_apps/deepstream-app \
	-I../eigen \
	-DDS_VERSION_MINOR=2 -DDS_VERSION_MAJOR=6

LIBS+= \
	-L$(CUDA_HOME)/lib64 -lcudart -lcuda \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_utils -lgstapp-1.0 \
	-lpthread -lm -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)

debug: CXXFLAGS += -DDEBUG -ggdb
debug: CCFLAGS += -DDEBUG -ggdb
debug: CFLAGS += -DDEBUG -ggdb
debug: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)



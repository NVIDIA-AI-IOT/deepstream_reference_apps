################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream-test5-analytics

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
DS_VER = $(shell deepstream-app -v | awk '$$1~/DeepStreamSDK/ {print substr($$2,1,3)}' )

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(DS_VER)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(DS_VER)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= deepstream_test5_app_main.c
SRCS+= ../deepstream-test5/deepstream_utc.c
SRCS+= ../deepstream-app/deepstream_app.c ../deepstream-app/deepstream_app_config_parser.c
SRCS+= $(wildcard ../../apps-common/src/*.c)

INCS= $(wildcard *.h)
INC_DIR=../deepstream-test5

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)
OBJS+= deepstream_nvdsanalytics_meta.o

CFLAGS+= -I../../apps-common/includes -I./includes -I../../../includes -I../deepstream-app/ -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5
CFLAGS+= -I$(INC_DIR)
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper \
       -lnvdsgst_customhelper -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm \
       -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

deepstream_nvdsanalytics_meta.o: deepstream_nvdsanalytics_meta.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -Wall -Werror $(CFLAGS) $<

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@  $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)


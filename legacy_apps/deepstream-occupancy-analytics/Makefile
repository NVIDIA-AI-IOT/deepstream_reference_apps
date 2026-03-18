################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/deepstream_utc.c
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app.c
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app_config_parser.c
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/src/*.c)
SRCS+= /opt/nvidia/deepstream/deepstream/sources/libs/nvds_msgapi_common_src/nvds_utils.cpp

INCS= $(wildcard *.h)
INC_DIR=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)
OBJS+= deepstream_nvdsanalytics_meta.o

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/apps/apps-common/includes -I./includes \
			-I/opt/nvidia/deepstream/deepstream/sources/includes \
			-I/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/ \
			-DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5
CFLAGS+= -I$(INC_DIR)
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper \
       -lnvdsgst_customhelper -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm \
       -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR) -lnvbufsurface -lnvds_logger -lcrypto
LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

deepstream_nvdsanalytics_meta.o: deepstream_nvdsanalytics_meta.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -Wall -Werror $(CFLAGS) $<

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@  $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -Wall -Werror $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)


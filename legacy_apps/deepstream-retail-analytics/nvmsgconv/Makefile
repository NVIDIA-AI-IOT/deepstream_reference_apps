###############################################################################
# SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
###############################################################################

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I./deepstream_schema

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -lyaml-cpp

SRCFILES:= nvmsgconv.cpp  \
  deepstream_schema/eventmsg_payload_peoplenet.cpp \
  deepstream_schema/dsmeta_payload.cpp \
  deepstream_schema/deepstream_schema.cpp
TARGET_LIB:= libnvds_msgconv.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)

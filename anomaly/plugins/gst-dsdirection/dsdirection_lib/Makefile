################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

PKGS:= gstreamer-1.0 gstreamer-base-1.0

DS_SDK_ROOT:=/opt/nvidia/deepstream/deepstream
CFLAGS+=-I $(DS_SDK_ROOT)/sources/includes
CFLAGS+=$(shell pkg-config --cflags $(PKGS))

all:
	gcc -ggdb $(CFLAGS) -c -o dsdirection_lib.o -fPIC dsdirection_lib.cpp
	ar rcs libdsdirection.a dsdirection_lib.o

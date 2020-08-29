/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DSDIRECTION_LIB__
#define __DSDIRECTION_LIB__
#include "nvll_osd_struct.h"
#include "nvds_opticalflow_meta.h"

#define MAX_LABEL_SIZE 128
#ifdef __cplusplus
extern "C"
{
#endif


// Detected/Labelled object structure, stores bounding box info along with label
typedef struct
{
  float flowx;
  float flowy;
  char direction[MAX_LABEL_SIZE];
} DsDirectionObject;

// Output data returned after processing
typedef struct
{
  DsDirectionObject object;
} DsDirectionOutput;

// Dequeue processed output
DsDirectionOutput *DsDirectionProcess (NvOFFlowVector * in_flow,
    int flow_cols, int flow_rows, int flow_bsize,
    NvOSD_RectParams * rect_param);


#ifdef __cplusplus
}
#endif

#endif

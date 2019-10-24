/**
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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

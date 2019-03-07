/*******************************************************************************
 * MIT License
 * 
 * Copyright (C) 2019 NVIDIA CORPORATION
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef __DSOPTICALFLOW_LIB__
#define __DSOPTICALFLOW_LIB__

/* Open CV headers */
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

typedef struct DsOpticalFlowCtx DsOpticalFlowCtx;

// Init parameters structure as input, required for instantiating dsexample_lib
typedef struct
{
  // Width at which frame/object will be scaled
  int processing_width;
  // height at which frame/object will be scaled
  int processing_height;
  // pool size
  int pool_size;
} DsOpticalFlowInitParams;

// Detected/Labelled object structure, stores bounding box info along with label
typedef struct
{
  int left;
  int top;
  int width;
  int height;
  char label[64];
} DsOpticalFlowObject;

// Output data returned after processing
typedef struct
{
  int rows;
  int cols;
  int elemSize;
  void* data;
  void* rgb;
} DsOpticalFlowOutput;

// Create the library context
DsOpticalFlowCtx * DsOpticalFlowCreate();

// Initialize library context
bool DsOpticalFlowCtxInit(DsOpticalFlowCtx *ctx,
                          DsOpticalFlowInitParams *init_params);

// Do frame Processing
bool DsOpticalFlowProcess(DsOpticalFlowCtx *ctx, void* nvbuf, 
                          int width, int height, DsOpticalFlowOutput* out);

void DsOpticalFlowFreeData(DsOpticalFlowCtx *ctx, void *data);

// Deinitialize library context
void DsOpticalFlowCtxDeinit (DsOpticalFlowCtx *ctx);

#endif

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

#include "dsdirection_lib.h"
#include <cstdio>
#include <cstdlib>
#include <bits/stdc++.h>
using namespace std;

#define PI 3.141592
#define PI_IN_DEGREES 180
//NVOF gives quarter-pixel output using OFSDK
#define FACTOR_QPEL 4.0


struct DsDirectionLabel
{
  //Start of interval in degrees
  float start;
  //End of interval in degrees
  float end;
  //Direction in UTF-8 Format
  const char *dirName;
};

static const DsDirectionLabel label[8] = {
  {157.5, -157.5, "\u21D0"},    //left
  {-67.5, -22.5, "\u21D8"},     //bottom-right
  {-112.5, -67.5, "\u21D3"},    //bottom
  {-157.5, -112.5, "\u21D9"},   //bottom-left
  {-22.5, 22.5, "\u21D2"},      //right
  {112.5, 157.5, "\u21D6"},     //top left
  {67.5, 112.5, "\u21D1"},      //top
  {22.5, 67.5, "\u21D7"}        //top-right
};

DsDirectionOutput *
DsDirectionProcess (NvOFFlowVector * in_flow, int flow_cols, int flow_rows,
    int flow_bsize, NvOSD_RectParams * rect_param)
{
  DsDirectionOutput *out =
      (DsDirectionOutput *) calloc (1, sizeof (DsDirectionOutput));
  //Mean for the whole object Bounding Box
  float x_mean = 0, y_mean = 0, max_radius = 0;
  //Frequency or no of pixels in the bounding box
  int freq_pix = 0;
  //Sum of Optical Flow value of pixels
  float x_mean_sum = 0, y_mean_sum = 0;

  //Get the motion inside the bbox. Sum it in x and y direction.
  for (unsigned int j = rect_param->top;
      j <= rect_param->top + rect_param->height; j++) {
    unsigned int block_j = j / flow_bsize;
    for (unsigned int i = rect_param->left;
        i <= rect_param->left + rect_param->width; i++) {
      // To get the mapping to the optical flow data (in_flow)
      unsigned int block_i = i / flow_bsize;
      unsigned int pos = block_j * flow_cols + block_i;

      //IF condition removes zero-motion pixels. thus getting a better estimate
      if ((in_flow[pos].flowx != 0) || in_flow[pos].flowy != 0) {
        x_mean_sum += (in_flow[pos].flowx) / FACTOR_QPEL;
        y_mean_sum += (in_flow[pos].flowy) / FACTOR_QPEL;
        freq_pix++;
      }
    }
  }

  //Final Mean
  x_mean = ((float) x_mean_sum) / freq_pix;
  y_mean = ((float) y_mean_sum) / freq_pix;
  out->object.flowx = x_mean;
  out->object.flowy = -y_mean;
  max_radius = sqrt (x_mean * x_mean + y_mean * y_mean);

  int dir = -1;
  //Assigning the direction based on optical flow data
  if (max_radius > 2) {
    //dir initialized to 0 to take care of special case for left direction.
    dir = 0;
    float angle = (atan2 (-y_mean, x_mean) * PI_IN_DEGREES) / PI;
    for (int i = 0; i <= 7; i++) {
      if (angle > label[i].start && angle <= label[i].end) {
        dir = i;
        break;
      }
    }
  }

  if (dir >= 0 && dir < 8) {
    snprintf (out->object.direction, 128, "%s", label[dir].dirName);
  } else {
    //Blank output when threshold is not crossed.
    snprintf (out->object.direction, 128, "%s", "");
  }

  return out;
}

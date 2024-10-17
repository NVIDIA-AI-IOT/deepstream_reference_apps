/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pair_graph.hpp"
#include "cover_table.hpp"
#include "munkres_algorithm.cpp"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

//#include "gstnvdsmeta.h"
//#include "gstnvdsinfer.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <stdio.h>
#include <vector>
#include <array>
#include <queue>
#include <cmath>

#define EPS 1e-6

template <class T>
using Vec1D = std::vector<T>;
template <class T>
using Vec2D = std::vector<Vec1D<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

static const int M = 2;

static Vec2D<int> topology{
    {0, 1, 15, 13},
    {2, 3, 13, 11},
    {4, 5, 16, 14},
    {6, 7, 14, 12},
    {8, 9, 11, 12},
    {10, 11, 5, 7},
    {12, 13, 6, 8},
    {14, 15, 7, 9},
    {16, 17, 8, 10},
    {18, 19, 1, 2},
    {20, 21, 0, 1},
    {22, 23, 0, 2},
    {24, 25, 1, 3},
    {26, 27, 2, 4},
    {28, 29, 3, 5},
    {30, 31, 4, 6},
    {32, 33, 17, 0},
    {34, 35, 17, 5},
    {36, 37, 17, 6},
    {38, 39, 17, 11},
    {40, 41, 17, 12}};

/* Method to find peaks in the output tensor. 'window_size' represents how many pixels we are considering at once to find a maximum value, or a ‘peak’. 
   Once we find a peak, we mark it using the ‘is_peak’ boolean in the inner loop and assign this maximum value to the center pixel of our window. 
   This is then repeated until we cover the entire frame. */
void find_peaks(Vec1D<int> &counts_out, Vec3D<int> &peaks_out, void *cmap_data,
                NvDsInferDims &cmap_dims, float threshold, int window_size, int max_count)
{
  int w = window_size / 2;
  int width = cmap_dims.d[2];
  int height = cmap_dims.d[1];

  counts_out.assign(cmap_dims.d[0], 0);
  peaks_out.assign(cmap_dims.d[0], Vec2D<int>(max_count, Vec1D<int>(M,
                                                                    0)));

  for (unsigned int c = 0; c < cmap_dims.d[0]; c++)
  {
    int count = 0;
    float *cmap_data_c = (float *)cmap_data + c * width * height;

    for (int i = 0; i < height && count < max_count; i++)
    {
      for (int j = 0; j < width && count < max_count; j++)
      {
        float value = cmap_data_c[i * width + j];

        if (value < threshold)
          continue;

        int ii_min = i - w;
        int jj_min = j - w;
        int ii_max = i + w + 1;
        int jj_max = j + w + 1;

        if (ii_min < 0)
          ii_min = 0;
        if (ii_max > height)
          ii_max = height;
        if (jj_min < 0)
          jj_min = 0;
        if (jj_max > width)
          jj_max = width;

        bool is_peak = true;
        for (int ii = ii_min; ii < ii_max; ii++)
        {
          for (int jj = jj_min; jj < jj_max; jj++)
          {
            if (cmap_data_c[ii * width + jj] > value)
            {
              is_peak = false;
            }
          }
        }

        if (is_peak)
        {
          peaks_out[c][count][0] = i;
          peaks_out[c][count][1] = j;
          count++;
        }
      }
    }

    counts_out[c] = count;
  }
}

/* Normalize the peaks found in 'find_peaks' and apply non-maximal suppression*/
Vec3D<float>
refine_peaks(Vec1D<int> &counts,
             Vec3D<int> &peaks, void *cmap_data, NvDsInferDims &cmap_dims,
             int window_size)
{
  int w = window_size / 2;
  int width = cmap_dims.d[2];
  int height = cmap_dims.d[1];

  Vec3D<float> refined_peaks(peaks.size(), Vec2D<float>(peaks[0].size(),
                                                        Vec1D<float>(peaks[0][0].size(), 0)));

  for (unsigned int c = 0; c < cmap_dims.d[0]; c++)
  {
    int count = counts[c];
    auto &refined_peaks_a_bc = refined_peaks[c];
    auto &peaks_a_bc = peaks[c];
    float *cmap_data_c = (float *)cmap_data + c * width * height;

    for (int p = 0; p < count; p++)
    {
      auto &refined_peak = refined_peaks_a_bc[p];
      auto &peak = peaks_a_bc[p];

      int i = peak[0];
      int j = peak[1];
      float weight_sum = 0.0f;

      for (int ii = i - w; ii < i + w + 1; ii++)
      {
        int ii_idx = ii;

        if (ii < 0)
          ii_idx = -ii;
        else if (ii >= height)
          ii_idx = height - (ii - height) - 2;

        for (int jj = j - w; jj < j + w + 1; jj++)
        {
          int jj_idx = jj;

          if (jj < 0)
            jj_idx = -jj;
          else if (jj >= width)
            jj_idx = width - (jj - width) - 2;

          float weight = cmap_data_c[ii_idx * width + jj_idx];
          refined_peak[0] += weight * ii;
          refined_peak[1] += weight * jj;
          weight_sum += weight;
        }
      }

      refined_peak[0] /= weight_sum;
      refined_peak[1] /= weight_sum;
      refined_peak[0] += 0.5;
      refined_peak[1] += 0.5;
      refined_peak[0] /= height;
      refined_peak[1] /= width;
    }
  }

  return refined_peaks;
}

/* Create a bipartite graph to assign detected body-parts to a unique person in the frame. This method also takes care of finding the line integral to assign scores
   to these points */
Vec3D<float>
paf_score_graph(void *paf_data, NvDsInferDims &paf_dims,
                Vec2D<int> &topology, Vec1D<int> &counts,
                Vec3D<float> &peaks, int num_integral_samples)
{
  int K = topology.size();
  int H = paf_dims.d[1];
  int W = paf_dims.d[2];
  int max_count = peaks[0].size();
  Vec3D<float> score_graph(K, Vec2D<float>(max_count, Vec1D<float>(max_count, 0)));

  for (int k = 0; k < K; k++)
  {
    auto &score_graph_nk = score_graph[k];
    auto &paf_i_idx = topology[k][0];
    auto &paf_j_idx = topology[k][1];
    auto &cmap_a_idx = topology[k][2];
    auto &cmap_b_idx = topology[k][3];
    float *paf_i = (float *)paf_data + paf_i_idx * H * W;
    float *paf_j = (float *)paf_data + paf_j_idx * H * W;

    auto &counts_a = counts[cmap_a_idx];
    auto &counts_b = counts[cmap_b_idx];
    auto &peaks_a = peaks[cmap_a_idx];
    auto &peaks_b = peaks[cmap_b_idx];

    for (int a = 0; a < counts_a; a++)
    {
      // Point A
      float pa_i = peaks_a[a][0] * H;
      float pa_j = peaks_a[a][1] * W;

      for (int b = 0; b < counts_b; b++)
      {
        // Point B
        float pb_i = peaks_b[b][0] * H;
        float pb_j = peaks_b[b][1] * W;

        // Vector from Point A to Point B
        float pab_i = pb_i - pa_i;
        float pab_j = pb_j - pa_j;

        // Normalized Vector from Point A to Point B
        float pab_norm = sqrtf(pab_i * pab_i + pab_j * pab_j) + EPS;
        float uab_i = pab_i / pab_norm;
        float uab_j = pab_j / pab_norm;

        float integral = 0.0;
        float increment = 1.0f / num_integral_samples;

        for (int t = 0; t < num_integral_samples; t++)
        {
          // Integral Point T
          float progress = (float)t / (float)num_integral_samples;
          float pt_i = pa_i + progress * pab_i;
          float pt_j = pa_j + progress * pab_j;

          // Convert to Integer
          int pt_i_int = (int)pt_i;
          int pt_j_int = (int)pt_j;

          // Edge cases for if the point is out of bounds, just skip them
          if (pt_i_int < 0)
            continue;
          if (pt_i_int > H)
            continue;
          if (pt_j_int < 0)
            continue;
          if (pt_j_int > W)
            continue;

          // Vector at integral point
          float pt_paf_i = paf_i[pt_i_int * W + pt_j_int];
          float pt_paf_j = paf_j[pt_i_int * W + pt_j_int];

          // Dot Product Normalized A->B with PAF Vector
          float dot = pt_paf_i * uab_i + pt_paf_j * uab_j;
          integral += dot;

          progress += increment;
        }

        // Normalize the integral with respect to the number of samples
        integral /= num_integral_samples;
        score_graph_nk[a][b] = integral;
      }
    }
  }
  return score_graph;
}

/*
 This method takes care of solving the graph assignment problem using Munkres algorithm. Munkres algorithm is defind in 'munkres_algorithm.cpp'
 */

Vec3D<int>
assignment(Vec3D<float> &score_graph,
           Vec2D<int> &topology, Vec1D<int> &counts, float score_threshold, int max_count)
{
  int K = topology.size();
  Vec3D<int> connections(K, Vec2D<int>(M, Vec1D<int>(max_count, -1)));

  Vec3D<float> cost_graph = score_graph;
  for (Vec2D<float> &cg_iter1 : cost_graph)
    for (Vec1D<float> &cg_iter2 : cg_iter1)
      for (float &cg_iter3 : cg_iter2)
        cg_iter3 = -cg_iter3;
  auto &cost_graph_out_a = cost_graph;

  for (int k = 0; k < K; k++)
  {
    int cmap_a_idx = topology[k][2];
    int cmap_b_idx = topology[k][3];
    int nrows = counts[cmap_a_idx];
    int ncols = counts[cmap_b_idx];
    auto star_graph = PairGraph(nrows, ncols);
    auto &cost_graph_out_a_nk = cost_graph_out_a[k];
    munkres_algorithm(cost_graph_out_a_nk, star_graph, nrows, ncols);

    auto &connections_a_nk = connections[k];
    auto &score_graph_a_nk = score_graph[k];

    for (int i = 0; i < nrows; i++)
    {
      for (int j = 0; j < ncols; j++)
      {
        if (star_graph.isPair(i, j) && score_graph_a_nk[i][j] > score_threshold)
        {
          connections_a_nk[0][i] = j;
          connections_a_nk[1][j] = i;
        }
      }
    }
  }
  return connections;
}

/* This method takes care of connecting all the body parts detected to each other 
   after finding the relationships between them in the 'assignment' method */
Vec2D<int>
connect_parts(
    Vec3D<int> &connections, Vec2D<int> &topology, Vec1D<int> &counts,
    int max_count)
{
  int K = topology.size();
  int C = counts.size();

  Vec2D<int> visited(C, Vec1D<int>(max_count, 0));

  Vec2D<int> objects(max_count, Vec1D<int>(C, -1));

  int num_objects = 0;
  for (int c = 0; c < C; c++)
  {
    if (num_objects >= max_count)
    {
      break;
    }

    int count = counts[c];

    for (int i = 0; i < count; i++)
    {
      if (num_objects >= max_count)
      {
        break;
      }

      std::queue<std::pair<int, int>> q;
      bool new_object = false;
      q.push({c, i});

      while (!q.empty())
      {
        auto node = q.front();
        q.pop();
        int c_n = node.first;
        int i_n = node.second;

        if (visited[c_n][i_n])
        {
          continue;
        }

        visited[c_n][i_n] = 1;
        new_object = true;
        objects[num_objects][c_n] = i_n;

        for (int k = 0; k < K; k++)
        {
          int c_a = topology[k][2];
          int c_b = topology[k][3];

          if (c_a == c_n)
          {
            int i_b = connections[k][0][i_n];
            if (i_b >= 0)
            {
              q.push({c_b, i_b});
            }
          }

          if (c_b == c_n)
          {
            int i_a = connections[k][1][i_n];
            if (i_a >= 0)
            {
              q.push({c_a, i_a});
            }
          }
        }
      }

      if (new_object)
      {
        num_objects++;
      }
    }
  }

  objects.resize(num_objects);
  return objects;
}

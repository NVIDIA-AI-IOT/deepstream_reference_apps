/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 32
#define THREADS_PER_BLOCK_1 (THREADS_PER_BLOCK - 1)

__global__ void
Convert_FtFTensorKernel(
        float *inBuffer,
        unsigned char *outBuffer,
        unsigned int width,
        unsigned int height)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        int v  = 255 * inBuffer[row * width + col];
        if(v < 0) {
          v = 0;
        } else if(v > 255) {
          v = 255;
        }
        outBuffer[row * width + col] = v;
    }
}

void
Convert_FtFTensor(
        float *inBuffer,
        unsigned char *outBuffer,
        unsigned int width,
        unsigned int height)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    Convert_FtFTensorKernel <<<blocks, threadsPerBlock, 0>>>
        (inBuffer, outBuffer, width, height);
}

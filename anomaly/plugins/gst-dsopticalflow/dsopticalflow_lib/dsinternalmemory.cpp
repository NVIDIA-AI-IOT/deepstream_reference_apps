////////////////////////////////////////////////////////////////////////////////
// MIT License
// 
// Copyright (C) 2019 NVIDIA CORPORATION
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include "dsinternalmemory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <pthread.h>

typedef struct _DsMemoryNode
{
    void *mem;
    struct _DsMemoryNode* next;
} MemoryNode;

struct _DsInternalMemoryPool
{
    MemoryNode*     buffer;
    MemoryNode*     used;
    uint32_t        n_bufs;
    uint32_t        bsize;
    bool            onGPU;
    pthread_mutex_t lock;
};

DsInternalMemoryPool *DsInternalMemoryPoolInit(uint32_t bsize, uint32_t num, bool onGPU)
{
    DsInternalMemoryPool *pool = (DsInternalMemoryPool*)malloc(sizeof(DsInternalMemoryPool));
    if (pool)
    {
        pool->buffer = NULL;
        pool->used = NULL;
        pool->bsize = bsize;
        pool->n_bufs = 0;
        pool->onGPU = onGPU;

        /* allocate the memory node based on the requested number */
        for (int i = 0; i < num; i++)
        {
            MemoryNode* node = (MemoryNode*)malloc(sizeof(MemoryNode));
            cudaError_t result;
            
            if (onGPU)
            {
                result = cudaMalloc(&node->mem, bsize);
            }
            else 
            {
                result = cudaMallocHost(&node->mem, bsize);
            }

            if (cudaSuccess != result)
            {
                fprintf(stderr, "Could not allocate cuda host buffer");
                break;
            }
            node->next = pool->buffer;
            pool->buffer = node;

            pool->n_bufs++;
        }

        pthread_mutex_init(&pool->lock, NULL);
    }

    return pool;
}

void *DsInternalMemoryPoolRequest(DsInternalMemoryPool* pool)
{
    void *mem = NULL;

    if (pool)
    {
        pthread_mutex_lock(&pool->lock);
        /* detach a node from pool and insert it to used */
        MemoryNode* node = pool->buffer;
        if (node)
        {
            pool->buffer = node->next;
            node->next = pool->used;
            pool->used = node;
            mem = node->mem;
        }
        pthread_mutex_unlock(&pool->lock);
    }

    return mem;
}

void DsInternalMemoryPoolReturn(DsInternalMemoryPool *pool, void *mem)
{
    if (pool)
    {
        pthread_mutex_lock(&pool->lock);
        /* search the memory node from used */
        MemoryNode* prev = NULL;
        MemoryNode* node = pool->used;
        while (node && node->mem != mem)
        {
            prev = node;
            node = node->next;
        }

        if (node)
        {
            if (prev == NULL)
            {
                pool->used = node->next;
            }
            else
            {
                prev->next = node->next;
            }

            /* move the memory node to the unused pool */
            node->next = pool->buffer;
            pool->buffer = node;
        }
        pthread_mutex_unlock(&pool->lock);
    }
}

void DsInternalMemoryPoolDeinit(DsInternalMemoryPool* pool)
{
    if (pool)
    {
        MemoryNode* curr = pool->buffer;
        while (curr)
        {
            MemoryNode* node = curr;
            curr = curr->next;

            if (pool->onGPU)
            {
                cudaFree(node->mem);
            }
            else 
            {
                cudaFreeHost(node->mem);
            }

            free(node);
        }

        pthread_mutex_lock(&pool->lock);
        curr = pool->used;
        while (curr)
        {
            MemoryNode* node = curr;
            curr = curr->next;

            if (pool->onGPU)
            {
                cudaFree(node->mem);
            }
            else
            {
                cudaFreeHost(node->mem);
            }

            free(node);
        }
        pthread_mutex_unlock(&pool->lock);
        pthread_mutex_destroy(&pool->lock);

        free(pool);
    }
}

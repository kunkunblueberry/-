#include <stdio.h>
#include <pthread.h>

// Use this code to time your threads
#include "CycleTimer.h"


/*

  15418 Spring 2012 note: This code was modified from example code
  originally provided by Intel.  To comply with Intel's open source
  licensing agreement, their copyright is retained below.

  -----------------------------------------------------------------

  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// Core computation of Mandelbrot set membershop
// Iterate complex number c to determine whether it diverges
static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

//
// MandelbrotSerial --
//
// Compute an image visualizing the mandelbrot set.  The resulting
// array contains the number of iterations required before the complex
// number corresponding to a pixel could be rejected from the set.
//
// * x0, y0, x1, y1 describe the complex coordinates mapping
//   into the image viewport.
// * width, height describe the size of the output image
// * startRow, endRow describe how much of the image to compute
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    unsigned int width, unsigned int height,
    int startRow, int endRow,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;//----这里处理很精细，是float，

    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);//-----------图像就是如此，二维其实本质上是一维
            output[index] = mandel(x, y, maxIterations);
        }
    }
}
/*
OpenCV 加载的图像，在内存中是连续的一维数组，但在逻辑上可以看作二维矩阵（行 × 列）。
假设图像是 height 行、width 列的灰度图：
每一行有 width 个像素，每个像素占 1 字节（因为是灰度图，仅需表示亮度）。
第 i 行第 j 列的像素，在内存中的一维索引为：i * width + j（第 0~i-1 行共占 i * width 个像素，再加上第 i 行的第 j 个像素）。

灰度图的每个像素用 1 个字节（unsigned char） 存储（范围 0~255，表示从黑到白的亮度）
*/
/*
以一个 24 英寸的显示器，分辨率为 1920×1080 为例，计算单个像素的物理大小：首先，24 英寸是指屏幕对角线的长度，根据勾股定理以及屏幕长宽比（常见为 16:9），
可以计算出屏幕的长和宽。假设屏幕长为 x 英寸，宽为 y 英寸，\(x^2 + y^2 = 24^2\)，且\(\frac{x}{y}=\frac{16}{9}\)，
解得 \(x \approx 20.9\) 英寸，\(y \approx 11.8\) 英寸。然后，水平方向单个像素的物理宽度约为 \(20.9 \div 1920 \approx 0.0109\) 英寸；
垂直方向单个像素的物理高度约为 \(11.8 \div 1080 \approx 0.0109\) 英寸
*/

// Struct for passing arguments to thread routine
typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;



//
// workerThreadStart --
//
// Thread entrypoint.
void* workerThreadStart(void* threadArgs) {

    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);

    // TODO: Implement worker thread here.
    float x0_s=args->x0;
    float x1_s=args->x1;
    float y0_s=args->y0;
    float y1_s=args->y1;
    unsigned int width_s=args->width;
    unsigned int height_s=args->height;
    int maxIterations_s=args->maxIterations;
    int *output_s=args->output;
    int threadId_s=args->threadId;
    int numthread=args->numThreads;
    
    printf("Hello world from thread %d\n", args->threadId);
    int rowsPerThread = (height_s+numthread-1)/ numthread; // 向上取整
    //平均分配其实并不可以达到最大加速，所以随着迭代次数增加，分配的线程增加才好
    //一个线程应该执行多大
    int start_row = threadId_s * rowsPerThread;
    int end_row=rowsPerThread+start_row;

    if((unsigned int)end_row>height_s){
        end_row=height_s;
    }
    mandelbrotSerial(x0_s,y0_s,x1_s,y1_s,width_s,height_s,start_row,end_row,maxIterations_s,output_s);
    
    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS||numThreads<=0)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1); 
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        // TODO: Set thread arguments here.
        args[i].threadId = i;
         // 关键：给每个线程的参数设置总线程数
        args[i].numThreads = numThreads;  // <-- 必须添加这一行
        // 同时确保其他参数（x0, y0, width, height等）也正确赋值
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}



/*
if(numthread==0)numthread=1;
    int rowsPerThread = height_s /( numthread+1);
     //一个线程应该执行多大
    int start_row = threadId_s* rowsPerThread;
    int end_row = start_row + rowsPerThread;
    if((unsigned int)(threadId_s+2)*rowsPerThread>width_s){
    start_row=(threadId_s+1)* rowsPerThread;
    end_row=width_s;
    }
能执行一点点，我觉得目前最大的问题是线程似乎可以为0，但是实际上线程数目是1
*/


/*
numthread++;

    int rowsPerThread = height_s/ numthread; // 向上取整
    //一个线程应该执行多大
    int start_row = threadId_s * rowsPerThread;
    int end_row=rowsPerThread+start_row;
        if((unsigned int)end_row>width_s){end_row=width_s;}

    mandelbrotSerial(x0_s, y0_s, x1_s, y1_s, width_s, height_s, start_row, end_row, maxIterations_s, output_s);
也不对
*/


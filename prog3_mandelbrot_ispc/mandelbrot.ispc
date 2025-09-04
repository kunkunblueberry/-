/*
[mandelbrot serial]:		[282.278] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[59.073] ms
Wrote image file mandelbrot-ispc.ppm
				(4.78x speedup from ISPC)

可以观察到加速比是4.78，不是八倍
*/

static inline int mandel(float c_re, float c_im, int count) {
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

export void mandelbrot_ispc(uniform float x0, uniform float y0, 
                            uniform float x1, uniform float y1,
                            uniform int width, uniform int height, 
                            uniform int maxIterations,
                            uniform int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    foreach (j = 0 ... height, i = 0 ... width) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = j * width + i;
            output[index] = mandel(x, y, maxIterations);
    }
}

// slightly different kernel to support tasking
task void mandelbrot_ispc_task(uniform float x0, uniform float y0, 
                               uniform float x1, uniform float y1,
                               uniform int width, uniform int height,
                               uniform int rowsPerTask,   //每个任务的行数
                               uniform int maxIterations,
                               uniform int output[])
{

    // taskIndex is an ISPC built-in
    表示当前任务的 ID（从 0 开始）。如果创建 4 个任务，taskIndex = 0,1,2,3。
    uniform int ystart = taskIndex * rowsPerTask;
    //计算一行要多少个任务
    uniform int yend = min(ystart + rowsPerTask, height);
    计算一列多少个任务

    乘积是一个块，就是最小执行单元
    
    uniform float dx = (x1 - x0) / width;
    uniform float dy = (y1 - y0) / height;
    
    foreach (j = ystart ... yend, i = 0 ... width) {
        j 是行（ystart 到 yend），i 是列（0 到 width）。
如果 ystart=0、yend=2、width=1，那么会并行处理以下所有 (j,i) 组合：
(0,0), (0,1),
(1,0), (1,1),
(2,0), (2,1)   //这样一个块是上面并行的大小

            float x = x0 + i * dx;
            float y = y0 + j * dy;
            
            int index = j * width + i;
            计算迭代次数并存储。
            output[index] = mandel(x, y, maxIterations);
    }
}

--使用task加速后达到了(7.55x speedup from task ISPC)


任务函数 process_chunk
task 修饰符：标记这是一个可被 launch 启动的任务函数，ISPC 编译器会为其生成适合多线程执行的代码。



export void mandelbrot_ispc_withtasks(uniform float x0, uniform float y0,
                                      uniform float x1, uniform float y1,
                                      uniform int width, uniform int height,
                                      uniform int maxIterations,
                                      uniform int output[])
{

    uniform int threadCount = 2;
    uniform int rowsPerTask = height / threadCount;
    划分成两段，一段一个cpu数

    // create threadCount tasks
    launch[threadCount] mandelbrot_ispc_task(x0, y0, x1, y1,
                                     width, height,
                                     rowsPerTask,
                                     maxIterations,
                                     output){
        if(height%threadCount!=0){    //处理多余的几行，也就是拿多余的用一个cpu跑罢了
         launch[1]mandelbrot_ispc_task( x0,  y0,
                                    t x1,  y1,
                                    width, height,
                                    maxIterations,
                                    output[])   
         task_sync;                                          
        }
    }
}

 uniform int threadCount = 2;
[mandelbrot serial]:		[291.368] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[55.664] ms
Wrote image file mandelbrot-ispc.ppm
[mandelbrot multicore ispc]:	[32.868] ms
Wrote image file mandelbrot-task-ispc.ppm
				(5.23x speedup from ISPC)
				(8.86x speedup from task ISPC)
线程为2有这样的数据


cpu为4
[mandelbrot serial]:		[221.547] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[44.060] ms
Wrote image file mandelbrot-ispc.ppm
[mandelbrot multicore ispc]:	[24.712] ms
Wrote image file mandelbrot-task-ispc.ppm
Mismatch : [748][175], Expected : 1, Actual : 0
Error : ISPC output differs from sequential output


[mandelbrot serial]:		[210.458] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[44.256] ms
Wrote image file mandelbrot-ispc.ppm
[mandelbrot multicore ispc]:	[18.440] ms
Wrote image file mandelbrot-task-ispc.ppm
Mismatch : [744][173], Expected : 1, Actual : 0
Error : ISPC output differs from sequential output
cpu为8

cpu为16
[mandelbrot serial]:		[221.384] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[44.186] ms
Wrote image file mandelbrot-ispc.ppm
[mandelbrot multicore ispc]:	[18.297] ms
Wrote image file mandelbrot-task-ispc.ppm
Mismatch : [736][168], Expected : 1, Actual : 0
Error : ISPC output differs from sequential output

调整一下核心（上面没调整虚拟机核心）
[mandelbrot serial]:		[276.599] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[40.007] ms
Wrote image file mandelbrot-ispc.ppm
[mandelbrot multicore ispc]:	[9.294] ms
Wrote image file mandelbrot-task-ispc.ppm
Mismatch : [736][168], Expected : 1, Actual : 0
Error : ISPC output differs from sequential output
//加速29.761倍，success


[mandelbrot serial]:		[278.368] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot ispc]:		[43.209] ms
Wrote image file mandelbrot-ispc.ppm
[mandelbrot multicore ispc]:	[8.694] ms
Wrote image file mandelbrot-task-ispc.ppm
Mismatch : [736][168], Expected : 1, Actual : 0
Error : ISPC output differs from sequential output
16核，加速到32倍



3. 附加分：（1分）Pthread抽象（在问题1中使用）与ISPC任务抽象之间有何区别？
（创建/连接）和（启动/同步）机制在语义上存在一些明显差异，
但这些差异的影响更为微妙。这里有一个思想实验来引导你作答：
当你启动10000个ISPC任务时会发生什么？当你启动10000个pthread时又会发生什么？
simd是以一块的方式分配到一个核的，foreach两个参数是一个块。
我觉得在并行层面有更小的损失（这里很像cuda的处理，不过这里是在cpu）
纯pthread有更大的损失


思维敏捷的学生提问：嘿，等一下！为什么向ISPC系统表达独立、
可并行化的工作有两种不同的机制（foreach和launch）呢？
系统难道不能将foreach的多次迭代分配到所有内核上，
同时为这些内核生成合适的SIMD代码吗？
回答：问得好！答案有很多种。我们在课堂上会进一步探讨。
不知道欸



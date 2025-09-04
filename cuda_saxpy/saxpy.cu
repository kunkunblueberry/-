#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

cudaError_t errorCheck(cudaError_t error_code, const char* fileName, int number) {
	if (error_code != cudaSuccess) {
		printf("cuda error:\r\n code=%d,name=%s,description=%s\r\nfile=%s,line=%d\r\n",
			error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), fileName, number);
	}
	return error_code;
}


void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;
    //这里分配的三倍区域，这个不能用！！！！，这是统一的，每一个区域只用三分之一

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    
    // TODO allocate device memory buffers on the GPU using cudaMalloc
    errorCheck(cudaMalloc((float**)&device_x,totalBytes/3),__FILE__,__LINE__);
    cudaMalloc((float**)&device_y,totalBytes/3);
    cudaMalloc((float**)&device_result,totalBytes/3);
    if(device_x!=NULL&&device_y!=NULL&&device_result!=NULL){
        cudaMemset(device_x,0,totalBytes/3);
        cudaMemset(device_y,0,totalBytes/3);
        cudaMemset(device_result,0,totalBytes/3);
    }else{
        printf("error to allocate memeory");
        exit(0);
    }

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO copy input arrays to the GPU using cudaMemcpy
    //cudaMemcpy(device_data, &host_data, sizeof(int), cudaMemcpyHostToDevice);主机到设备
    //cudaMemcpy(fpHost_C, ipDevice_C, ByteCount, cudaMemcpyDeviceToHost)  设备到主机
    errorCheck(cudaMemcpy(device_x,xarray,totalBytes/3,cudaMemcpyHostToDevice),__FILE__,__LINE__);
    errorCheck(cudaMemcpy(device_y,yarray,totalBytes/3,cudaMemcpyHostToDevice),__FILE__,__LINE__);
    

    // run kernel
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    errorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__);

    //
    // TODO copy result from GPU using cudaMemcpy
    errorCheck(cudaMemcpy(resultarray,device_result,totalBytes/3,cudaMemcpyDeviceToHost),__FILE__,__LINE__);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();//API 函数，用于获取 “最近一次 CUDA 操作产生的错误码”（即使之前没有显式检查错误，也能捕获到
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    // TODO free memory buffers on the GPU
    errorCheck(cudaFree(device_x),__FILE__,__LINE__);
    cudaFree(device_y);
    cudaFree(device_result);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
/*
问题：比较并解释两组定时器（你添加的定时器和提供的起始代码中已有的定时器）给出的结果之间的差异。
观察到的带宽值与机器不同组件所报告的可用带宽大致相符吗？
提示：你应该通过网络查找NVIDIA RTX 2080 GPU的内存带宽，以及计算机PCIe-x16总线的最大传输速度。
它是PCIe 3.0，一条16通道的总线，用于连接CPU和GPU。

这里主要就是想说明内存带宽这方面是否有严重的性能偏差，带宽有没有诡异的东西影响
*/


/*
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce RTX 4050 Laptop GPU
   SMs:        20
   Global mem: 6140 MB
   CUDA Cap:   8.9
---------------------------------------------------------
Overall: 106.346 ms             [2.102 GB/s]
Overall: 30.237 ms              [7.392 GB/s]
Overall: 30.284 ms              [7.381 GB/s]
*/

/*
1,Makefile因为是自己本地机器，所以makefile里面的架构要改，包括cuda版本和对应的gpu架构
cuda很看重这个，gpu等等各种版本，一定都要对应上

2，函数上面的主要错误是分配区域越界，分配的total是总共的，每个device只用三分之一

3，对于核函数的任何操作都应该只使用cuda版本，这里初始化memset也应该是cudaMemset

4,cuda的gdb是单独的，要使用的话要在注册表打开
*/
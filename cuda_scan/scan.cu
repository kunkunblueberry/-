#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* 辅助函数，用于向上取整到2的幂。
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
// __global__ void kernel_A(int*A,int length,int depth){
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int iters = std::min(blockDim.x, length / depth);
//     //这么想，d作为步长，最大是iters，所以这个是一步,就是小的规约块，一个多大
//     //depth是 “分层系数”（每个分层包含的元素数量，由 CPU 端传入，比如 32）。
    

//     //一个线程块里面循环规约到一个线程块的尽头
//     //这里还要理解，i是每一个线程块都执行操作，不是某一个
//     for(int d=1;d<iters;d<<=1){
//         if(i<length/depth&&i%(2*d)==0){    //到这里还是在处理一个线程块里面的东西
//             int idx=i*depth;        //----这个通过倍数关系普吉岛多个线程块
//             A[idx]=A[idx]+A[idx-d];     //----观察几何关系，每次等于加上前面第1，2，4，8个数字
//         }
//         __syncthreads();
//     }    
// }
// /*
// 这么想的，一个无穷长度，但是2^n长度，分配给给线程块计算，一个线程块里面循环规约到尽头
// 外在来看，还剩下需要规约的就剩下n/block.dim个，这里就可以外部进行一次循环，把剩下的规约了
// */

// __global__ void kernel_B(int*A,int length,int depth){
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int iters = std::min(blockDim.x, length / depth);

//     for(int d=iters/2;d>=1;d/=2){
//         if(i<length/depth&&i%(2*d)==0){
//             int idx=i*depth;            //----第一轮下，i是0，N刚好越界
//             int t=A[idx+d-1];
//             A[idx+d-1]=A[idx+2*d-1];
//             A[idx+2*d-1]+=t;
//         }
//         __syncthreads();
//     }
// }

// __global__ void kernel_c(int *A,int length){
//     extern __shared__ int sdata[];
//     int tid=threadIdx.x;
//     int i=blockIdx.x * blockDim.x + threadIdx.x;

//     sdata[tid]=(i<n)?A[i]:0;

//     __syncthreads();

//     for(int s=1;s<blockDim.x;s<<=1){
//         int index=2*s*tid;
//          if (index + s < blockDim.x) {
//             sdata[index]=sdata[index]+sdata[index+d];
//         }
//         __syncthreads();
//     }
// }
// __global__ void kernel_A(int *device_data,int length){
//     __shared__  int shared[length];

//     int *input_begin =device_data+blockDim.x*blockIdx.x;
//     shared[threadIdx.x]=input_begin[threadIdx.x];
//     __syncthreads();

//     //明确index是在原始数据上要处理的位置
//     //threadIdx.x是避免线程分化而用的聚在一块，一部分线程，要通过几何关系映射出数据
//     if(threadIdx.x<blockDim.x/2)
//     {
//         int index1=threadIdx.x*2+1;     //--------这个只是决定用几号线程去处理它
//         shared[index1]+=shared[index1-1];
//     }
//     //序号不对，先处理第一行这一批
//     __syncthreads();
    
//     for(int i=2;i<blockDim.x;i*=2){
//         if(threadIdx.x<blockDim.x/(2*i)){
//             int index=threadIdx.x*2*i;      //--------这个只是决定用几号线程去处理它
//             shared[index+i+1]+=shared[index+i-1];
//         }
//         __syncthreads();
//     }
//     //这个和视频演示代码的难点是，这是反过来的
//     device_data[threadIdx.x]=shared[threadIdx.x];
// }


//泪目，终于写对了
__global__ void kernel_A(int* device_data, int length) {
	extern __shared__  int shared[];
	int n =blockDim.x * blockIdx.x + threadIdx.x;
	shared[threadIdx.x] =(n<length)? device_data[n]:0;
	__syncthreads();
	
	for (int i = 1; i < blockDim.x; i *= 2) {
		if (threadIdx.x < blockDim.x / (2 * i)) {
			int index = (2 * i * (threadIdx.x+1) - 1);// 2 * (i - 1)*(threadIdx.x+1) + 1;
			shared[index] += shared[index - i];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	shared[blockDim.x - 1] = 0;

	__syncthreads();

	for (int i = blockDim.x / 2; i >0; i/=2) {
		int index = 2 * i * (threadIdx.x + 1) - 1;
		if (index<blockDim.x) {
			int t = shared[index-i];
			shared[index-i] = shared[index];
			shared[index] += t;
			}
		__syncthreads();
	}
	if(n<length)
	device_data[n] = shared[threadIdx.x];
}



void exclusive_scan(int* device_data, int length)
{
    /* 待办事项
     * 用你的并行前缀和实现填充这个函数。
     * 向你传递了设备内存中数据的位置
     * 数据已初始化为输入值。你的代码应该
     
     * 执行原地扫描，在同一个数组中生成结果。
     * 这是主机代码——你需要声明一个或多个CUDA
     * 内核（使用__global__修饰符）以在GPU上并行运行代码。
     * 注意，给了你数组的实际长度，但可以假设
     * 数据数组的大小足以容纳比输入大的下一个2的幂。
     */
    //基本任务是实现数组长度2^n情况下的前缀和
    //要做到极限速度应该就使用共享内存和线程块级别的规约以及调用核函数规约

    //这里有好几种方法，根据长度可以看是线程块规约还是可以选择直接来，不过直接来始终是要慢一点
    int L=nextPow2(length);
    
    const int threadsPerBlock = 512;
    const int blocks = (L + threadsPerBlock - 1) / threadsPerBlock;

    kernel_A<<<blocks, threadsPerBlock,threadsPerBlock*sizeof(int)>>>(device_data,L);
    cudaDeviceSynchronize();

    cudaError_t errCode = cudaPeekAtLastError();//API 函数，用于获取 “最近一次 CUDA 操作产生的错误码”（即使之前没有显式检查错误，也能捕获到
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}

/* 这个函数是你将要编写的代码的包装器——它将输入复制到GPU并计时上面的exclusive_scan()函数的调用。你不应该修改它。
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // 我们将数组大小向上取整到2的幂，但原始输入结束后的元素未初始化且不检查正确性。
    // 在你的实现中，如果假设数组长度是2的幂可能会更容易，但这会导致对非2的幂输入做额外工作。
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // 等待所有剩余工作完成。
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Thrust库并行前缀和函数的包装器
 * 与上面类似，将输入复制到GPU并仅计时扫描本身的执行。
 * 不期望你的实现性能能与Thrust版本竞争。
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}



int find_peaks(int *device_input, int length, int *device_output) {
    /* 待办事项：
     * 找出列表中所有大于前后元素的元素，
     * 将元素的索引存储到device_result中。
     * 返回找到的峰值元素数量。
     * 根据定义，第0个元素和第length-1个元素都不是峰值。
     *
     * 你的任务是实现这个函数。你可能想要
     * 使用一个或多个对exclusive_scan()的调用，以及
     * 额外的CUDA内核启动。
     * 注意：与扫描代码一样，我们确保分配的数组大小是2的幂，所以如果需要的话，你可以在它们上使用你的exclusive_scan函数。但是，你必须确保给定原始长度时find_peaks的结果是正确的。
     */

    int _length = nextPow2(length);
    exclusive_scan(device_input,_length);
    cudaDeviceSynchronize();
    int *result=malloc(_length*sizeof(int));
    int *output=malloc(_length*sizeof(int));
    
    cudaMemcpy(result,device_input,_length*sizeof(int),cudaMemcpyDeviceToHost);
    int count=0;
    for(int i=1;i<_length-1;i++){
        if(result[i]>result[i-1]&&result[i]>result[i+1]){
            output[count]=i;
            count++;
        }
    }
    cudaMemcpy(device_output,output,_length*sizeof(int),cudaMemcpyHostToDevice);   
    return count;
}
/*
死活写不对，不过这里也可以知道一些要点咯
1，设备指针只能在核函数操作！！！！
2，一些计算的细节
*/



/* find_peaks的计时包装器。你不应该修改这个函数。
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),     //这个结果是索引数组，但是是设备函数，所以意思似乎是让我再写一个内核函数
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // 为了有趣，只打印出机器上的一些统计信息

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("找到 %d 个CUDA设备\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("设备 %d: %s\n", i, deviceProps.name);
        printf("   流处理器:        %d\n", deviceProps.multiProcessorCount);
        printf("   全局内存: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA计算能力:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

/*
为什么要取min(blockDim.x, length/depth)？
iters的作用是限制线程块内归约迭代的最大步长（即d的上限）。为什么要用最小值？
因为归约的迭代步长d不能超过两个限制：
不能超过线程块的线程数量（blockDim.x）
归约的迭代逻辑是 “步长d每次翻倍”（d=1→2→4→...），而每次迭代中，线程块内需要有足够的线程来处理所有分层。如果d超过blockDim.x，会导致线程索引超出线程块的范围（比如线程块只有 16 个线程，d=32时，i+d会超过线程数量，导致无线程处理）。
不能超过当前的分层总数（length/depth）
如果分层总数本身比线程块的线程数少（比如分层总数 = 8，线程块有 16 个线程），那么迭代步长d最多只能到分层总数的一半（否则i+d会超出分层范围）。此时iters需要被限制为分层总数，避免线程处理不存在的分层。
*/

/*
举个具体例子，让iters的作用更清晰
假设：
数组长度length=1024
当前depth=32（每个分层 32 个元素）→ 分层总数length/depth=32
线程块大小blockDim.x=16（每个线程块 16 个线程）
则iters = min(16, 32) = 16。
这个16意味着：
线程块内的归约迭代中，步长d最大只能到8（因为d从 1 开始翻倍：1→2→4→8，下一次d=16时会超过iters=16，循环停止）。
为什么要限制到d=8？
因为线程块只有 16 个线程，最多能处理 “步长d=8” 的迭代：此时每个线程处理的分层索引i范围是 0~15（16 个线程），i+d的范围是 8~23，都在分层总数 32 的范围内，且线程数量足够覆盖所有操作。
*/
/*
length/depth（这里是 8）本质上是 “分层索引的最大值 + 1”（因为分层索引是 0~7）。当d的取值使得i + d >= length/depth时，(i + d) * depth必然会超过数组的最大索引（length-1=255），导致越界。
而iters = min(blockDim.x, length/depth)的作用就是通过限制d的最大迭代值（d < iters），确保i + d始终小于length/depth（即分层索引不越界），最终保证数组索引不越界。
*/

/*
关于 10000 长的数组如何处理。因为一个线程块的线程数是有限的（比如常见的 256 或 512），10000 不能被一个块处理，所以需要多个线程块组成网格。
每个块处理一部分数据，比如块大小为 256，那么需要的块数是 ceil (10000/256) = 40 个块（因为 256*39=9984，剩下 16 个元素由第 40 块处理）

线程的执行范围（哪些线程处理哪些数据）
核函数reduce2的执行范围由网格（Grid）和线程块（Block）的配置决定。在调用核函数时，需要指定块数和每块的线程数，例如：
reduce2<<<gridDim, blockDim>>>(d_x, d_y);  // gridDim是块数，blockDim是每块线程数


第一步：确定需要多少个线程块？
总元素数N = 10000，每块处理blockDim.x = 256个元素，因此需要的块数为：
gridDim.x = ceil(N / blockDim.x) = ceil(10000 / 256) = 40（因为 256×39=9984，剩下 16 个元素需要第 40 块处理）。
此时，核函数调用为：reduce2<<<40, 256>>>(d_x, d_y);
*/
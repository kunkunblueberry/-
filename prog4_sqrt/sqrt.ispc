
static const float kThreshold = 0.00001f; 

export void sqrt_ispc(uniform int N,
                      uniform float initialGuess,
                      uniform float values[],
                      uniform float output[])
{
    foreach (i = 0 ... N) {

        float x = values[i];
        //对N个数据进行这样的操作，这里就是N个数据并行

        float guess = initialGuess;//这应该是假设值，后面要进行迭代，
        //迭代成功了就是真实值

        float pred = abs(guess * guess * x - 1.f);

        while (pred > kThreshold) {  //这里是迭代差距，要小于0.000001
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            pred = abs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;。。传出答案
        
    }
}


//下面两个是一套函数
task void sqrt_ispc_task(uniform int N,
                         uniform int span,
                         uniform float initialGuess,
                         uniform float values[],
                         uniform float output[])
{

    uniform int indexStart = taskIndex * span;
    //span 表示每个并行任务负责处理的元素数量。

    uniform int indexEnd = min(N, indexStart + span);
    //非整除数据处理，取小的，最后launch一个核处理一组数据就行了啊
    
    foreach (i = indexStart ... indexEnd) {

        float x = values[i];
        float guess = initialGuess;

        float pred = abs(guess * guess * x - 1.f);

        while (pred > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            pred = abs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    
    }
}

export void sqrt_ispc_withtasks(uniform int N,
                                uniform float initialGuess,
                                uniform float values[],
                                uniform float output[])
{

    uniform int span = N / 64;  // 64 tasks

    launch[N/span] sqrt_ispc_task(N, span, initialGuess, values, output);

    if(N%span!=0){
        launch[1] sqrt_ispc_task(N, span, initialGuess, values, output);
    }
    sync;
}







单一数据并行（SIMD 并行）
export void sqrt_ispc(...) {
    foreach (i = 0 ... N) {  // 直接遍历所有元素
        // 对每个 i 执行迭代计算
    }
}
仅使用 foreach 实现单层次的 SIMD 数据并行。
ISPC 编译器会将循环转换为 SIMD 指令，在单个硬件线程内，
通过多个 SIMD 通道（如 16 个通道）同时处理多个 i 对应的元素
（例如一次处理 16 个元素）



任务并行 + 数据并行（两层并行）
task void sqrt_ispc_task(...) {  // task 修饰符：标记为可并行任务
    // 1. 确定当前任务处理的子范围
    uniform int indexStart = taskIndex * span;  // taskIndex：当前任务 ID
    uniform int indexEnd = min(N, indexStart + span);
    
    // 2. 在子范围内用 foreach 做数据并行
    foreach (i = indexStart ... indexEnd) {
        // 对每个 i 执行迭代计算
    }
}。
外层：通过 launch 启动多个 sqrt_ispc_task 任务
（每个任务由独立硬件线程执行），即多线程任务并行。
每个任务通过 taskIndex 确定自己处理的数据块（indexStart 到 indexEnd），
实现数据分块处理。

内层：每个任务内部用 foreach 对自己负责的子范围做SIMD 数据并行（
单个线程内的向量并行）。



task1

[sqrt serial]:		[1102.131] ms
[sqrt ispc]:		[202.727] ms
[sqrt task ispc]:	[16.850] ms
				(5.44x speedup from ISPC)
				(65.41x speedup from task ISPC)


- **无任务ISPC**：单线程内用SIMD指令并行，加速依赖向量宽度，受单核心内存带宽限制。  
- **有任务ISPC**：多线程分块+SIMD，加速更高，受全局内存带宽限制（多核共享带宽）。  
- **核心瓶颈**：内存带宽（非浮点算力），优化后可接近但难达线性加速比。

saxpy 操作（z[i] = a*x[i] + y[i]）是典型的内存密集型计算（而非计算密集型）
每个元素的计算仅包含 1 次乘法和 1 次加法（2 次浮点运算）
但需要 3 次内存访问（读 x[i]、读 y[i]、写 z[i]），因此性能瓶颈主要是内存带宽，而非浮点运算能力。
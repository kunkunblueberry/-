#include <algorithm>

// Generate random data
void initRandom(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
    }
}

// Generate data that gives high relative speedup
void initGood(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // Todo: Choose values
        //思考，加速比（并行比上串行）高，要么是串行慢，要么是并行快

        values[i] = 2.999999f;
    }
}

// Generate data that gives low relative speedup
void initBad(float *values, int N) {
    values[0]=0.999999;
    for (int i=0; i<N; i++)
    {
        // Todo: Choose values
        int sign = (rand() % 2 == 0) ? 1 : -1;  // 50% 概率 1，50% 概率 -1
        values[i+1] = values[i] * sign;
    }
}

/*
values[i] = 2.999999f;
获得了如此大的加速

[sqrt serial]:		[4072.976] ms
[sqrt ispc]:		[602.129] ms
[sqrt task ispc]:	[51.014] ms
				(6.76x speedup from ISPC)
				(79.84x speedup from task ISPC)

                [sqrt serial]:		[4524.612] ms

[sqrt ispc]:		[641.849] ms
[sqrt task ispc]:	[50.333] ms
				(7.05x speedup from ISPC)
				(89.89x speedup from task ISPC)
这次甚至接近90，说明2.999999有极大的加速比
参考图片
在1.0附近只有个位数的加速
                */

/*
取1.0
[sqrt serial]:		[24.803] ms
[sqrt ispc]:		[11.196] ms
[sqrt task ispc]:	[6.647] ms
				(2.22x speedup from ISPC)
				(3.73x speedup from task ISPC)

                [sqrt serial]:		[23.138] ms
[sqrt ispc]:		[12.998] ms
[sqrt task ispc]:	[7.104] ms
				(1.78x speedup from ISPC)
				(3.26x speedup from task ISPC)

*/
/*
从3.0->1.5，加速比都是下降趋势（只做了一组，但是差距比较大，没有重复）

数学上面建模一下
*/
/*
void initBad(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // Todo: Choose values
        values[i] = .000001f+0.999998f* static_cast<float>(rand()) / RAND_MAX;
    }
}
    这他妈反而干到60+加速比了

到0.5从个位数加速比又到31.94了

0.0000001又
[sqrt serial]:		[2303.061] ms
[sqrt ispc]:		[333.938] ms
[sqrt task ispc]:	[30.097] ms
				(6.90x speedup from ISPC)
				(76.52x speedup from task ISPC)
总的来说刚好呈现一个核迭代次数正相关的图像
*/

/*

要计算这段代码的串行比例（\(\alpha\)），需要先明确：串行比例是指程序中 “必须单线程执行、无法并行化” 的代码占总执行时间的比例。我们可以从代码结构和执行流程入手，拆分串行部分和并行部分的时间占比。一、代码执行流程拆解先梳理 sqrt_ispc_withtasks 的完整执行过程（假设 N 是总数据量，且足够大）：串行初始化阶段（主线程单线程执行）：计算 span = N / 64（确定每个任务的块大小）。计算任务数量：N/span = 64（因为 span = N/64）。判断 N%span != 0（检查是否需要额外任务处理剩余数据）。并行任务启动（主线程发起，任务并行执行）：启动 64 个任务（launch[N/span]），每个任务处理 span 个元素。若有余数，再启动 1 个任务处理剩余元素（最多 span-1 个）。并行计算阶段（多线程 + SIMD 并行）：每个任务（sqrt_ispc_task）内部：计算 indexStart 和 indexEnd（每个任务的处理范围，uniform 变量，单线程内串行计算）。通过 foreach 对 [indexStart, indexEnd] 范围内的元素进行 SIMD 并行计算（核心逻辑，耗时最长）。串行同步阶段（主线程等待）：sync：主线程等待所有启动的任务执行完毕（阻塞等待，单线程状态）。二、串行部分与并行部分的划分1. 串行部分（必须单线程执行的代码）这些操作无法被并行化，只能由主线程（或单个任务线程）依次执行，构成串行比例的核心：串行操作说明耗时特性初始化计算span = N / 64、N/span、N%span 判断等算术运算。耗时极短（微秒级甚至纳秒级），与 N 大小几乎无关。launch 任务调度启动 64 个（+1 个）任务的系统调用、参数传递、线程分配等。耗时较短（与任务数量正相关，64 个任务的调度开销通常在毫秒级以内）。sync 同步等待主线程等待所有任务完成的阻塞时间（理论上为 0，实际有微小同步开销）。几乎不消耗计算资源，主要受任务执行时间影响（任务越快完成，等待时间越短）。任务内 uniform 变量计算每个任务内 indexStart、indexEnd 的计算（单线程内串行）。每个任务内耗时极短，且不同任务的这部分计算可并行，因此不计入全局串行时间。2. 并行部分（可多线程 / SIMD 加速的代码）这是代码的核心耗时部分，可被并行化：任务并行：64 个任务被分配到 8 核 CPU 上并行执行（多线程并行）。数据并行：每个任务内部通过 foreach 进行 SIMD 指令并行（单个线程内的向量加速）。耗时特性：总耗时与 N 成正比（处理每个元素需要迭代计算 while (pred > kThreshold)），是代码的主要耗时部分（占总时间的 90% 以上，甚至更高）。三、串行比例（\(\alpha\)）的估算假设总执行时间为 \(T_{\text{total}}\)，串行部分时间为 \(T_{\text{serial}}\)，并行部分时间为 \(T_{\text{parallel}}\)，则：
\(T_{\text{total}} = T_{\text{serial}} + T_{\text{parallel}}\)
\(\alpha = \frac{T_{\text{serial}}}{T_{\text{total}}}\)具体估算（以大规模 N 为例）：设 \(N = 1,000,000\)（足够大，确保并行部分主导耗时）。假设单个元素的迭代计算耗时 \(t = 1\ \mu s\)（1 微秒），则总计算量为 \(1,000,000 \times t = 1000\ ms\)（并行部分理论耗时，不考虑加速）。8 核 CPU 并行 + SIMD 加速后，假设实际并行耗时 \(T_{\text{parallel}} \approx 1000 / (8 \times 4) = 31.25\ ms\)（假设 SIMD 再加速 4 倍）。串行部分耗时 \(T_{\text{serial}}\)：初始化（0.1 ms）+ 任务调度（0.5 ms）+ 同步（0.1 ms）≈ 0.7 ms。则总时间：
\(T_{\text{total}} = 0.7 + 31.25 = 31.95\ ms\)串行比例：
\(\alpha = \frac{0.7}{31.95} \approx 2.19\%\)四、关键结论串行比例极低：对于大规模数据（N 很大），串行部分（初始化、任务调度、同步）的耗时占比通常低于 5%，甚至可低至 1% 以下。加速比主要受核心数限制：根据阿姆达尔定律，当 \(\alpha \approx 2\%\) 时，8 核 CPU 的理论加速比为：
\(S = \frac{1}{\alpha + \frac{1-\alpha}{8}} \approx \frac{1}{0.02 + \frac{0.98}{8}} \approx 6.7\ \text{倍}\)
（接近 8 核的理想加速比，因为串行部分占比低）。当前代码的潜在问题：任务数量（64 个）远大于 CPU 核心数（8 核），会导致频繁的线程切换开销，反而增加串行调度时间（提高 \(\alpha\)）。建议将任务数设为与核心数一致（如 8 个），span = N / 8，减少调度开销，进一步降低串行比例。总结：这段代码的串行比例主要由任务调度和初始化开销决定，在大规模数据下约为 1% - 5%，理论加速比接近 CPU 核心数（8 核下约 6 - 7 倍）。优化任务数量可进一步提高加速比。*/
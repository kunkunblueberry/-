// 在此处定义向量单元宽度
#define VECTOR_WIDTH 8

#ifndef CMU418INTRIN_H_
#define CMU418INTRIN_H_

#include <cstdlib>
#include <cmath>
#include "logger.h"

//*******************
//* 类型定义 *
//*******************

extern Logger CMU418Logger;

template <typename T>
//这是一个通用模板结构体，用于表示向量寄存器（存储多个相同类型的数据，模拟 SIMD 指令中的向量）
struct __cmu418_vec {
  T value[VECTOR_WIDTH];
  //当T为float且VECTOR_WIDTH=8时，__cmu418_vec<float>
  //表示一个包含 8 个单精度浮点数的向量寄存器，可同时对 8 个浮点数执行并行操作
};

// 用__cmu418_mask声明一个掩码
struct __cmu418_mask : __cmu418_vec<bool> {};

// 用__cmu418_vec_float声明一个浮点型向量寄存器
#define __cmu418_vec_float __cmu418_vec<float>

// 用__cmu418_vec_int声明一个整型向量寄存器
#define __cmu418_vec_int   __cmu418_vec<int>

//***********************
//* 函数定义 *
//***********************

// 返回一个掩码，其前N个通道初始化为1，其余通道初始化为0
__cmu418_mask _cmu418_init_ones(int first = VECTOR_WIDTH);
//__cmu418_mask mask = _cmu418_init_ones(3); // 结果：[1,1,1,0,0,0,0,0]

/*
比如数组长度是 10（VECTOR_WIDTH=8），分两批处理：
第一批处理前 8 个（全有效）；
第二批处理剩下的 2 个（只有前 2 个有效，后 6 个无效）。
这时候第二批的掩码就是_cmu418_init_ones(2)（[1,1,0,0,0,0,0,0]），确保只处理有效元素，避免越界错误。
*/

// 返回maska的反掩码
__cmu418_mask _cmu418_mask_not(__cmu418_mask &maska);
 
// 返回(maska | maskb)的结果掩码
__cmu418_mask _cmu418_mask_or(__cmu418_mask &maska, __cmu418_mask &maskb);

// 返回(maska & maskb)的结果掩码
__cmu418_mask _cmu418_mask_and(__cmu418_mask &maska, __cmu418_mask &maskb);

// 计算掩码maska中1的数量
int _cmu418_cntbits(__cmu418_mask &maska);

// 如果向量通道处于激活状态，则将寄存器设置为指定值；否则保持其旧值
void _cmu418_vset_float(__cmu418_vec_float &vecResult, float value, __cmu418_mask &mask);
void _cmu418_vset_int(__cmu418_vec_int &vecResult, int value, __cmu418_mask &mask);

// 为方便用户，返回一个所有通道都初始化为指定值的向量寄存器
__cmu418_vec_float _cmu418_vset_float(float value);
__cmu418_vec_int _cmu418_vset_int(int value);

// 如果向量通道处于激活状态，将向量寄存器src中的值复制到向量寄存器dest；否则保持dest的旧值
void _cmu418_vmove_float(__cmu418_vec_float &dest, __cmu418_vec_float &src, __cmu418_mask &mask);
void _cmu418_vmove_int(__cmu418_vec_int &dest, __cmu418_vec_int &src, __cmu418_mask &mask);

// 如果向量通道处于激活状态，将数组src中的值加载到向量寄存器dest；否则保持dest的旧值
void _cmu418_vload_float(__cmu418_vec_float &dest, float* src, __cmu418_mask &mask);
void _cmu418_vload_int(__cmu418_vec_int &dest, int* src, __cmu418_mask &mask);

// 如果向量通道处于激活状态，将向量寄存器src中的值存储到数组dest；否则保持dest的旧值
void _cmu418_vstore_float(float* dest, __cmu418_vec_float &src, __cmu418_mask &mask);
void _cmu418_vstore_int(int* dest, __cmu418_vec_int &src, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca + vecb)的计算结果；否则保持旧值
void _cmu418_vadd_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vadd_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca - vecb)的计算结果；否则保持旧值
void _cmu418_vsub_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vsub_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca * vecb)的计算结果；否则保持旧值
void _cmu418_vmult_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vmult_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca / vecb)的计算结果；否则保持旧值
void _cmu418_vdiv_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vdiv_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca >> vecb)的计算结果；否则保持旧值
void _cmu418_vshiftright_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca & vecb)的计算结果；否则保持旧值
void _cmu418_vbitand_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回veca的绝对值abs(veca)；否则保持旧值
void _cmu418_vabs_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_mask &mask);
void _cmu418_vabs_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca > vecb)的掩码结果；否则保持旧值
void _cmu418_vgt_float(__cmu418_mask &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vgt_int(__cmu418_mask &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca < vecb)的掩码结果；否则保持旧值
void _cmu418_vlt_float(__cmu418_mask &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vlt_int(__cmu418_mask &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 如果向量通道处于激活状态，返回(veca == vecb)的掩码结果；否则保持旧值
void _cmu418_veq_float(__cmu418_mask &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_veq_int(__cmu418_mask &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// 对相邻元素对求和，例如：
// [0 1 2 3] -> [0+1 0+1 2+3 2+3]
void _cmu418_hadd_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &vec);

// 执行奇偶交错操作，所有偶数索引的元素移到数组的前半部分，奇数索引的元素移到后半部分，例如：
// [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
void _cmu418_interleave_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &vec);

// 添加自定义日志以辅助调试
void addUserLog(const char * logStr);

#endif
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	if (x < 0) {
	    output[i] = -x;
	} else {
	    output[i] = x;
	}
    }
}

// 使用15418内部函数实现绝对值
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    // 注意：仔细看一下这个循环的索引。这个示例代码
    // 不能保证在(N % VECTOR_WIDTH) != 0的情况下正常工作。
    // 这是为什么呢？
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// 掩码全为1
	maskAll = _cmu418_init_ones();

	// 掩码全为0
	maskIsNegative = _cmu418_init_ones(0);

	// 从连续的内存地址加载值向量
	_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

	// 根据条件设置掩码
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// 使用掩码执行指令（"if"分支）
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// 对maskIsNegative取反以生成"else"掩码
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// 执行指令（"else"分支）
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// 将结果写回内存
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// 接收一个值数组和一个指数数组
// 对于每个元素，计算values[i]^exponents[i]并将值限制在
// 4.18。将结果存储在输出中。
// 使用迭代平方，因此总迭代次数与指数的log_2成比例
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	float result = 1.f;
	int y = exponents[i];
	float xpower = x;

	while (y > 0) {
	    if (y & 0x1) {  //判断奇偶性，奇数为1
			result *= xpower;
		}
	    xpower = xpower * xpower;
	    y >>= 1;//右移，也就是除以2并且向下取整
	}
	if (result > 4.18f) {
	    result = 4.18f;
	}
	output[i] = result;
    }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // 在这里实现clampedExpSerial的向量化版本
    __cmu418_vec_float x;	//声明一个x的向量寄存器，
	__cmu418_vec_int y;	//
	__cmu418_vec_float result=_cmu418_vset_float(1.f);  //存储结果
	__cmu418_vec_float xpower;
	//寄存器对应上面的四个初始值声明

	__cmu418_vec_int zero = _cmu418_vset_int(0);//用于判断数据大于小于0
	__cmu418_vec_int one = _cmu418_vset_int(1);//用于左移操作
	__cmu418_vec_float mid = _cmu418_vset_float(4.18f);//用于加载操作
	__cmu418_vec_int temp=_cmu418_vset_int(0);
	//循环开始
	int i=0;
	for(i=0;i + VECTOR_WIDTH <= N;i+=VECTOR_WIDTH){
	__cmu418_mask maskAll, maskIsNe, maskIsNotNegative, active_mask, mask_over4;
	//声明四个掩码；maskIsNegative表示是奇数
	maskIsNe = _cmu418_init_ones(0);
	maskAll = _cmu418_init_ones();
	active_mask=_cmu418_init_ones(0);
	mask_over4=_cmu418_init_ones(0);
	
	//把values的值加载到x里面
	_cmu418_vload_float(x, values+i, maskAll);  //这里是对应的一般情况，不是末尾
	_cmu418_vload_int(y, exponents+i, maskAll);//加载y进入
	_cmu418_vload_float(xpower, values+i, maskAll);//xpower的加载

	_cmu418_vgt_int(active_mask,y, zero, maskAll);
	//这一步代替while循环，和0进行操作，大于零的才有一位掩码，才能进行后续操作

	//这里很困难啊，不清楚如何准确摘出里面大于0的y进行操作。
	_cmu418_vgt_int(active_mask,y, zero, maskAll);
	while(_cmu418_cntbits(active_mask)!=0){
	
	_cmu418_vbitand_int(temp, y,one, active_mask);
	//这里对active的进行操作，进一步找到里面是奇数的
	_cmu418_veq_int(maskIsNe,temp,one,active_mask);
	_cmu418_vmult_float(result,xpower,result,maskIsNe);//乘法操作

	//else情况
	maskIsNotNegative = _cmu418_mask_not(maskIsNe);
	_cmu418_vmult_float(xpower,xpower,xpower,maskIsNotNegative);
	_cmu418_vshiftright_int(y,y,one,maskIsNotNegative);//右移操作

	}
	_cmu418_vgt_float(mask_over4,result,mid,maskAll);//if (result > 4.18f)
	_cmu418_vset_float(result,4.18f,mask_over4);

	_cmu418_vstore_float(output+i, result, maskAll);//返回输出
	_cmu418_vgt_int(active_mask,y,zero,maskAll);
}

	//掩码声明对不满足8的整数倍的数据操作
		int extar=N%VECTOR_WIDTH;
	if(extar>0){
		__cmu418_mask mask=_cmu418_init_ones(extar);
		//把values的值加载到x里面
		__cmu418_mask  maskIsNe, maskIsNotNegative,active_mask,mask_over4;
		//声明四个掩码；
		__cmu418_vec_float x_tail, result_tail, xpower_tail;
    	__cmu418_vec_int y_tail;
		result_tail = _cmu418_vset_float(1.0f);
		__cmu418_vec_int temp;
		//声明尾部的几个数字
	maskIsNe = _cmu418_init_ones(0);
	active_mask=_cmu418_init_ones(0);
	mask_over4=_cmu418_init_ones(0);
	
	_cmu418_vload_float(x_tail, values+i, mask);  //这里是对应的一般情况，不是末尾
	_cmu418_vload_int(y_tail, exponents+i, mask);//加载y进入
	_cmu418_vload_float(xpower_tail, values+i, mask);//xpower的声明

	_cmu418_vgt_int(active_mask, zero, y_tail,mask);//找出大于0
	while(_cmu418_cntbits(active_mask)){

	_cmu418_vbitand_int(temp,y_tail,one,active_mask);//if (y & 0x1)
	_cmu418_veq_int(maskIsNe,temp,one,active_mask);
	_cmu418_vmult_float(result_tail,xpower_tail,result_tail,maskIsNe);//乘法操作
	maskIsNotNegative = _cmu418_mask_not(maskIsNe);
	
	//else情况
	_cmu418_vmult_float(xpower_tail,xpower_tail,xpower_tail,maskIsNotNegative);
	_cmu418_vshiftright_int(y_tail,y_tail,one,maskIsNotNegative);//右移操作
	}
	_cmu418_vgt_float(mask_over4,result_tail,mid,mask);//if (result > 4.18f)
	_cmu418_vset_float(result_tail,4.18f,mask_over4);

	_cmu418_vstore_float(output+i,result_tail,mask);
	}
//后半部分处理不是8的倍数，也就是最后临界的几个数字
}


float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
	sum += values[i];
    }

    return sum;
}

// 假设N % VECTOR_WIDTH == 0
// 假设VECTOR_WIDTH是2的幂
float arraySumVector(float* values, int N) {
    // 在这里实现你的向量化版本
	// _cmu418_vset_float result;
    // for(int i=0;i+VECTOR_WIDTH<N;i+=2*VECTOR_WIDTH){
	// 	_cmu418_vset_float x;
	// 	_cmu418_vset_float y;
	// 	__cmu418_mask maskAll;
	// 	maskAll=_cmu418_init_ones();
	// 	_cmu418_vload_float(x,values+i,maskAll);
	// 	_cmu418_vload_float(y,values+i+VECTOR_WIDTH,maskAll);
	// 	_cmu418_vadd_float(result,x,y,maskAll);
	// }
	return 0.f;
}
/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */



#include "hipSYCL/sycl/libkernel/sscp/builtins/reduction.hpp"

#define SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_reduce_##type(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::plus>(x); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::multiply>(x); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::min>(x); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::max>(x); \
        default: \
            return __acpp_##type{}; \
    } \
} \

SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(f16)
SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(f32)
SUBGROUP_FLOAT_SUB_GROUP_REDUCTION(f64)

#define SUBGROUP_INT_SUB_GROUP_REDUCTION(fn_suffix,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::plus>(x); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::multiply>(x); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::min>(x); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::max>(x); \
        case __acpp_sscp_algorithm_op::bit_and: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::bit_and>(x); \
        case __acpp_sscp_algorithm_op::bit_or: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::bit_or>(x); \
        case __acpp_sscp_algorithm_op::bit_xor: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::bit_xor>(x); \
        case __acpp_sscp_algorithm_op::logical_and: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::logical_and>(x); \
        case __acpp_sscp_algorithm_op::logical_or: \
            return __acpp_reduce_over_subgroup<__acpp_sscp_algorithm_op::logical_or>(x); \
        default: \
            return __acpp_##type{}; \
    } \
} \

SUBGROUP_INT_SUB_GROUP_REDUCTION(i8 ,int8 )
SUBGROUP_INT_SUB_GROUP_REDUCTION(i16,int16)
SUBGROUP_INT_SUB_GROUP_REDUCTION(i32,int32)
SUBGROUP_INT_SUB_GROUP_REDUCTION(i64,int64)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u8 ,uint8 )
SUBGROUP_INT_SUB_GROUP_REDUCTION(u16,uint16)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u32,uint32)
SUBGROUP_INT_SUB_GROUP_REDUCTION(u64,uint64)


#define SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_work_group_reduce_##type(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::plus>(x); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::multiply>(x); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::min>(x); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::max>(x); \
        default: \
            return __acpp_##type{}; \
    } \
} \

SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(f16)
SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(f32)
SUBGROUP_FLOAT_WORK_GROUP_REDUCTION(f64)

#define SUBGROUP_INT_WORK_GROUP_REDUCTION(fn_suffix,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_work_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::plus>(x); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::multiply>(x); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::min>(x); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::max>(x); \
        case __acpp_sscp_algorithm_op::bit_and: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::bit_and>(x); \
        case __acpp_sscp_algorithm_op::bit_or: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::bit_or>(x); \
        case __acpp_sscp_algorithm_op::bit_xor: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::bit_xor>(x); \
        case __acpp_sscp_algorithm_op::logical_and: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::logical_and>(x); \
        case __acpp_sscp_algorithm_op::logical_or: \
            return __acpp_reduce_over_work_group<__acpp_sscp_algorithm_op::logical_or>(x); \
        default: \
            return __acpp_##type{}; \
    } \
} \

SUBGROUP_INT_WORK_GROUP_REDUCTION(i8 ,int8 )
SUBGROUP_INT_WORK_GROUP_REDUCTION(i16,int16)
SUBGROUP_INT_WORK_GROUP_REDUCTION(i32,int32)
SUBGROUP_INT_WORK_GROUP_REDUCTION(i64,int64)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u8 ,uint8 )
SUBGROUP_INT_WORK_GROUP_REDUCTION(u16,uint16)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u32,uint32)
SUBGROUP_INT_WORK_GROUP_REDUCTION(u64,uint64)

// #include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu/ockl.hpp"


// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_int8 __acpp_sscp_sub_group_reduce_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_int16 __acpp_sscp_sub_group_reduce_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_int32 __acpp_sscp_sub_group_reduce_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_int64 __acpp_sscp_sub_group_reduce_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_uint8 __acpp_sscp_sub_group_reduce_u8(__acpp_sscp_algorithm_op op, __hipsycl_uint8 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_uint16 __acpp_sscp_sub_group_reduce_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_uint32 __acpp_sscp_sub_group_reduce_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_uint64 __acpp_sscp_sub_group_reduce_u64(__acpp_sscp_algorithm_op op, __hipsycl_uint64 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_f16 __acpp_sscp_sub_group_reduce_f16(__acpp_sscp_algorithm_op op, __hipsycl_f16 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_f32 __acpp_sscp_sub_group_reduce_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x){
//     switch(op)
//     {
//         case __acpp_sscp_algorithm_op::plus:
//             return __ockl_wfred_add_f32(x); 
//         case __acpp_sscp_algorithm_op::multiply:
//             break;
//         case __acpp_sscp_algorithm_op::min:
//             return __ockl_wfred_min_f32(x); 
//         case __acpp_sscp_algorithm_op::max:
//             return __ockl_wfred_max_f32(x); 
//         case __acpp_sscp_algorithm_op::bit_and:
//             break;
//         case __acpp_sscp_algorithm_op::bit_or:
//             break;
//         case __acpp_sscp_algorithm_op::bit_xor:
//             break;
//         case __acpp_sscp_algorithm_op::logical_and:
//             break;
//         case __acpp_sscp_algorithm_op::logical_or:
//             break;
//         default:
//             break;
//     }
// }



// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_f64 __acpp_sscp_sub_group_reduce_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);



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

#include "hipSYCL/sycl/libkernel/sscp/builtins/scan_inclusive.hpp"

#define SUBGROUP_FLOAT_REDUCTION(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_inclusive_scan_##type(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, plus{}); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, multiply{}); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, min{}); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, max{}); \
    } \
} \

SUBGROUP_FLOAT_REDUCTION(f16)
SUBGROUP_FLOAT_REDUCTION(f32)
SUBGROUP_FLOAT_REDUCTION(f64)

#define SUBGROUP_INT_REDUCTION(fn_suffix,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_inclusive_scan_##fn_suffix(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, plus{}); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, multiply{}); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, min{}); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, max{}); \
        case __acpp_sscp_algorithm_op::bit_and: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, bit_and{}); \
        case __acpp_sscp_algorithm_op::bit_or: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, bit_or{}); \
        case __acpp_sscp_algorithm_op::bit_xor: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, bit_xor{}); \
        case __acpp_sscp_algorithm_op::logical_and: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, logical_and{}); \
        case __acpp_sscp_algorithm_op::logical_or: \
            return __acpp_subgroup_inclusive_scan_impl_##type(x, logical_or{}); \
    } \
} \

SUBGROUP_INT_REDUCTION(i8 ,int8 )
SUBGROUP_INT_REDUCTION(i16,int16)
SUBGROUP_INT_REDUCTION(i32,int32)
SUBGROUP_INT_REDUCTION(i64,int64)
SUBGROUP_INT_REDUCTION(u8 ,uint8 )
SUBGROUP_INT_REDUCTION(u16,uint16)
SUBGROUP_INT_REDUCTION(u32,uint32)
SUBGROUP_INT_REDUCTION(u64,uint64)

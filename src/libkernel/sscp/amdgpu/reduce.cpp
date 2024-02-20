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
#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu/ockl.hpp"


// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_int8 __hipsycl_sscp_sub_group_reduce_i8(__hipsycl_sscp_algorithm_op op, __hipsycl_int8 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_int16 __hipsycl_sscp_sub_group_reduce_i16(__hipsycl_sscp_algorithm_op op, __hipsycl_int16 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_int32 __hipsycl_sscp_sub_group_reduce_i32(__hipsycl_sscp_algorithm_op op, __hipsycl_int32 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_int64 __hipsycl_sscp_sub_group_reduce_i64(__hipsycl_sscp_algorithm_op op, __hipsycl_int64 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_uint8 __hipsycl_sscp_sub_group_reduce_u8(__hipsycl_sscp_algorithm_op op, __hipsycl_uint8 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_uint16 __hipsycl_sscp_sub_group_reduce_u16(__hipsycl_sscp_algorithm_op op, __hipsycl_uint16 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_uint32 __hipsycl_sscp_sub_group_reduce_u32(__hipsycl_sscp_algorithm_op op, __hipsycl_uint32 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_uint64 __hipsycl_sscp_sub_group_reduce_u64(__hipsycl_sscp_algorithm_op op, __hipsycl_uint64 x);

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_f16 __hipsycl_sscp_sub_group_reduce_f16(__hipsycl_sscp_algorithm_op op, __hipsycl_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f32 __hipsycl_sscp_sub_group_reduce_f32(__hipsycl_sscp_algorithm_op op, __hipsycl_f32 x){
    switch(op)
    {
        case __hipsycl_sscp_algorithm_op::plus:
            return __ockl_wfred_add_f32(x); 
        case __hipsycl_sscp_algorithm_op::multiply:
            break;
        case __hipsycl_sscp_algorithm_op::min:
            return __ockl_wfred_min_f32(x); 
        case __hipsycl_sscp_algorithm_op::max:
            return __ockl_wfred_max_f32(x); 
        case __hipsycl_sscp_algorithm_op::bit_and:
            break;
        case __hipsycl_sscp_algorithm_op::bit_or:
            break;
        case __hipsycl_sscp_algorithm_op::bit_xor:
            break;
        case __hipsycl_sscp_algorithm_op::logical_and:
            break;
        case __hipsycl_sscp_algorithm_op::logical_or:
            break;
        default:
            break;
    }
}

// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __hipsycl_f64 __hipsycl_sscp_sub_group_reduce_f64(__hipsycl_sscp_algorithm_op op, __hipsycl_f64 x);
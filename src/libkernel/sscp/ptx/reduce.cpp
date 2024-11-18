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

__acpp_uint64 get_active_mask(){
    __acpp_uint64 subgroup_size = __acpp_sscp_get_subgroup_size();
    return (1ull << subgroup_size)-1;
}


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_work_group_reduce_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x){
  const constexpr __acpp_uint32 shmem_array_length = 32;
  static __attribute__((loader_uninitialized))  __attribute__((address_space(3))) int shrd_mem[shmem_array_length];

  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       subgroup_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_uint32       first_sg_size = __acpp_sscp_get_subgroup_size();
  first_sg_size = __acpp_sscp_work_group_broadcast_i32(0, first_sg_size);

  const __acpp_uint32       block_reduction_iteration_increment = subgroup_size <= shmem_array_length ? subgroup_size : shmem_array_length;

  const __acpp_uint32       num_subgroups = (wg_size+subgroup_size-1)/subgroup_size;
  const __acpp_uint32       subgroup_id = wg_lid/subgroup_size;
  const __acpp_uint32       sg_lid    = __acpp_sscp_get_subgroup_id_

  __acpp_f32 local_reduce_result = __acpp_sscp_sub_group_reduce_f32(op, x);
  //Sum up until all sgs can load their data into shmem
  if(subgroup_id < shmem_array_length){
    shrd_mem[subgroup_id] = local_reduce_result;
  }
   __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  
  for(int i = shmem_array_length; i < num_subgroups; i+= shmem_array_length){
    if(subgroup_id > i && suprgoup_id < shmem_array_length){
         shrd_mem[subgroup_id] += local_reduce_result;
         __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
    }
  }
  // Now we are filled up shared memory with the results of all the subgroups
  __
  // We reduce in shared memory until it fits into one sg
  for(int i = shmem_array_length/2; i > first_sg_size; i /= 2){
    if(wg_lid < i){
      shrd_mem[wg_lid] += shrd_mem[wg_lid+i];
    }
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  // Now we load the data into the first element and do a final reduction and a broadcast at the end
  if(wg_lid < first_sg_size){
    local_reduce_result = shrd_mem[wg_lid];
    local_reduce_result =  __acpp_sscp_sub_group_reduce_f32(op, x);
  }
  local_reduce_result = __acpp__acpp_sscp_sub_group_broadcast_i32(0, local_reduce_result);
  return local_reduce_result;
}

#define SUBGROUP_FLOAT_REDUCTION(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_reduce_##type(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_reduce_over_group_##type(x, plus{}); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_reduce_over_group_##type(x, multiply{}); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_reduce_over_group_##type(x, min{}); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_reduce_over_group_##type(x, max{}); \
    } \
} \

SUBGROUP_FLOAT_REDUCTION(f32)
SUBGROUP_FLOAT_REDUCTION(f64)

#define SUBGROUP_INT_REDUCTION(fn_suffix,type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##type __acpp_sscp_sub_group_reduce_##fn_suffix(__acpp_sscp_algorithm_op op, __acpp_##type x){ \
    switch(op) \
    { \
        case __acpp_sscp_algorithm_op::plus: \
            return __acpp_reduce_over_group_##type(x, plus{}); \
        case __acpp_sscp_algorithm_op::multiply: \
            return __acpp_reduce_over_group_##type(x, multiply{}); \
        case __acpp_sscp_algorithm_op::min: \
            return __acpp_reduce_over_group_##type(x, min{}); \
        case __acpp_sscp_algorithm_op::max: \
            return __acpp_reduce_over_group_##type(x, max{}); \
        case __acpp_sscp_algorithm_op::bit_and: \
            return __acpp_reduce_over_group_##type(x, bit_and{}); \
        case __acpp_sscp_algorithm_op::bit_or: \
            return __acpp_reduce_over_group_##type(x, bit_or{}); \
        case __acpp_sscp_algorithm_op::bit_xor: \
            return __acpp_reduce_over_group_##type(x, bit_xor{}); \
        case __acpp_sscp_algorithm_op::logical_and: \
            return __acpp_reduce_over_group_##type(x, logical_and{}); \
        case __acpp_sscp_algorithm_op::logical_or: \
            return __acpp_reduce_over_group_##type(x, logical_or{}); \
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
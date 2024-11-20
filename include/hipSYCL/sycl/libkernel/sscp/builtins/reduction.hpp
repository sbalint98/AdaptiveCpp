/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "builtin_config.hpp"
#include "core_typed.hpp"
#include "broadcast.hpp"
#include "utils.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

#ifndef HIPSYCL_SSCP_REDUCTION_BUILTINS_HPP
#define HIPSYCL_SSCP_REDUCTION_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_reduce_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_reduce_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_reduce_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_reduce_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_work_group_reduce_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_work_group_reduce_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_work_group_reduce_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_work_group_reduce_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_work_group_reduce_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_work_group_reduce_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_work_group_reduce_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);





HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_reduce_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_reduce_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_reduce_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_reduce_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_sub_group_reduce_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_sub_group_reduce_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_sub_group_reduce_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_sub_group_reduce_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_sub_group_reduce_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_sub_group_reduce_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_sub_group_reduce_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);




// reduce
template <typename OutType, typename BinaryOperation> 
OutType __acpp_reduce_over_subgroup_impl(OutType x, BinaryOperation binary_op, __acpp_int32 active_threads) {
  const __acpp_uint32       lrange     = __acpp_sscp_get_subgroup_max_size(); 
  const __acpp_uint32       lid        = __acpp_sscp_get_subgroup_local_id(); 
  const __acpp_uint64       subgroup_size = active_threads; 
  auto local_x = x; 
  for (__acpp_int32 i = lrange / 2; i > 0; i /= 2) {  
    auto other_x=bit_cast<OutType>(__acpp_sscp_sub_group_select(bit_cast<typename integer_type<OutType>::type>(local_x), lid+i)); 
    if (lid +i < subgroup_size) 
        local_x = binary_op(local_x, other_x); 
    } 
    return bit_cast<OutType>(__acpp_sscp_sub_group_select(bit_cast<typename integer_type<OutType>::type>(local_x), 0)); 
} 


template < __acpp_sscp_algorithm_op binary_op, typename OutType> 
OutType __acpp_reduce_over_subgroup(OutType x) {
  using op = typename get_op<binary_op>::type;
  const __acpp_uint32       lrange     = __acpp_sscp_get_subgroup_max_size(); 
  return __acpp_reduce_over_subgroup_impl(x, op{}, lrange);
} 

template <typename OutType, typename BinaryOperation> 
OutType __acpp_reduce_over_work_group_impl(OutType x,BinaryOperation op){
  const constexpr __acpp_uint32 shmem_array_length = 32;
  static __attribute__((loader_uninitialized))  __attribute__((address_space(3))) int shrd_mem[shmem_array_length];

  const __acpp_uint32       wg_lid     = __acpp_sscp_typed_get_local_linear_id<3, int>();
  const __acpp_uint32       wg_size    = __acpp_sscp_typed_get_local_size<3, int>();
  const __acpp_uint32       max_sg_size = __acpp_sscp_get_subgroup_max_size();
  const __acpp_uint32       sg_size = __acpp_sscp_get_subgroup_size();
  const __acpp_uint32 first_sg_size = __acpp_sscp_work_group_broadcast_i32(0, sg_size);

  const __acpp_uint32       block_reduction_iteration_increment = max_sg_size <= shmem_array_length ? max_sg_size : shmem_array_length;

  const __acpp_uint32       num_subgroups = (wg_size+max_sg_size-1)/max_sg_size;
  const __acpp_uint32       subgroup_id = wg_lid/max_sg_size;
  const __acpp_uint32       sg_lid    = __acpp_sscp_get_subgroup_id();


  __acpp_f32 local_reduce_result = __acpp_reduce_over_subgroup_impl(x, op, max_sg_size);
  
  // return local_reduce_result;
  //Sum up until all sgs can load their data into shmem
  if(subgroup_id < shmem_array_length){
    shrd_mem[subgroup_id] = local_reduce_result;
  }
   __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

   
  for(int i = shmem_array_length; i < num_subgroups; i+= shmem_array_length){
    if(subgroup_id > i && subgroup_id < shmem_array_length){
         shrd_mem[subgroup_id] += local_reduce_result;
         __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
    }
  }

  // Now we are filled up shared memory with the results of all the subgroups
  // We reduce in shared memory until it fits into one sg
  for(int i = shmem_array_length/2; i > first_sg_size; i /= 2){
    if(wg_lid < i){
      shrd_mem[wg_lid] += shrd_mem[wg_lid+i];
    }
  }

  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  // Now we load the data into the first element

  
  if(wg_lid < first_sg_size){
    local_reduce_result = shrd_mem[wg_lid];
    int active_threads = num_subgroups < first_sg_size ? num_subgroups : first_sg_size;
    local_reduce_result =  __acpp_reduce_over_subgroup_impl(local_reduce_result, plus{}, active_threads);
  }
  
  // Do a final broadcast
  local_reduce_result = bit_cast<float>(__acpp_sscp_work_group_broadcast_i32(0, bit_cast<__acpp_uint32>(local_reduce_result)));
  return local_reduce_result;
}

template < __acpp_sscp_algorithm_op binary_op, typename OutType> 
OutType __acpp_reduce_over_work_group(OutType x) {
  using op = typename get_op<binary_op>::type;
  return __acpp_reduce_over_work_group_impl(x, op{});
} 
#endif

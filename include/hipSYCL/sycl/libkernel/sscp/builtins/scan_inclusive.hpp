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
#include "utils.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

#ifndef HIPSYCL_SSCP_SCAN_INCLUSIVE_BUILTINS_HPP
#define HIPSYCL_SSCP_SCAN_INCLUSIVE_BUILTINS_HPP

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_inclusive_scan_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_inclusive_scan_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_inclusive_scan_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_inclusive_scan_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_sub_group_inclusive_scan_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_sub_group_inclusive_scan_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_sub_group_inclusive_scan_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_sub_group_inclusive_scan_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_sub_group_inclusive_scan_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_sub_group_inclusive_scan_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_sub_group_inclusive_scan_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);

// reduce
#define SCAN_OVER_SUBGROUP(outType,size) \
template <typename T, typename BinaryOperation> \
T __acpp_subgroup_inclusive_scan_impl_##outType (T x, BinaryOperation binary_op) { \
  const __acpp_uint32       lid        = __acpp_sscp_get_subgroup_local_id(); \
  const __acpp_uint32       lrange     = __acpp_sscp_get_subgroup_max_size(); \
  const __acpp_uint64       subgroup_size = __acpp_sscp_get_subgroup_size(); \
  auto local_x = x; \
  for (__acpp_int32 i = 1; i < lrange; i *= 2) {  \
    __acpp_uint32 next_id = lid -i; \
    auto other_x=bit_cast<__acpp_##outType>(__acpp_sscp_sub_group_select_i##size(bit_cast<__acpp_uint##size>(local_x), next_id)); \
    if (next_id >= 0 && i <= lid) \
        local_x = binary_op(local_x, other_x); \
    } \
    return local_x; \
} \

SCAN_OVER_SUBGROUP(f16,16)
SCAN_OVER_SUBGROUP(f32,32)
SCAN_OVER_SUBGROUP(f64,64)
SCAN_OVER_SUBGROUP(int8,8)
SCAN_OVER_SUBGROUP(int16,16)
SCAN_OVER_SUBGROUP(int32,32)
SCAN_OVER_SUBGROUP(int64,64)
SCAN_OVER_SUBGROUP(uint8,8)
SCAN_OVER_SUBGROUP(uint16,16)
SCAN_OVER_SUBGROUP(uint32,32)
SCAN_OVER_SUBGROUP(uint64,64)



// // inclusive_scan
// template <typename T, typename BinaryOperation>
// __device__ T __acpp_inclusive_scan_over_group(sub_group g, T x,
//                                                  BinaryOperation binary_op) {
//   const size_t       lid        = g.get_local_linear_id();
//   const size_t       lrange     = g.get_local_linear_range();
//   const unsigned int activemask = __ballot_sync(0xffffffff, 1);

//   auto local_x = x;

//   for (size_t i = 1; i < lrange; i *= 2) {
//     size_t next_id = lid - i;
//     if (i > lid)
//       next_id = 0;

//     auto other_x = detail::__acpp_shuffle_impl(local_x, next_id);
//     if (activemask & (1 << (next_id)) && i <= lid && lid < lrange)
//       local_x = binary_op(local_x, other_x);
//   }

//   return local_x;
// }

#endif
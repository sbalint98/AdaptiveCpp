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



#if __has_builtin(__builtin_bit_cast)

#define HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, in, out)                           \
  out = __builtin_bit_cast(Tout, in)

#else

#define HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, in, out)                           \
  {                                                                            \
    union {                                                                    \
      Tout union_out;                                                          \
      Tin union_in;                                                            \
    } u;                                                                       \
    u.union_in = in;                                                           \
    out = u.union_out;                                                         \
  }
#endif

template <class Tout, class Tin>
Tout bit_cast(Tin x) {
  Tout result;
  HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, x, result);
  return result;
}

__acpp_uint64 get_active_mask();


struct plus
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs+rhs;
    }
};

struct min
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs < rhs ? lhs : rhs;
    }
};

struct max
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs < rhs ? rhs : lhs;
    }
};

struct multiply
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs < rhs ? rhs : lhs;
    }
};

struct bit_and
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs & rhs ;
    }
};

struct bit_or
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs | rhs;
    }
};

struct bit_xor
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs ^ rhs;
    }
};

struct logical_and
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs and rhs;
    }
};

struct logical_or
{
    template<typename T>
    T operator()(T lhs, T rhs){
        return lhs or rhs;
    }
};
// reduce
#define REDUCE_OVER_GROUP(outType,size) \
template <typename T, typename BinaryOperation> \
T __acpp_reduce_over_group_##outType (T x, BinaryOperation binary_op) { \
  const __acpp_uint32       lid        = __acpp_sscp_get_subgroup_local_id(); \
  const __acpp_uint32       lrange     = __acpp_sscp_get_subgroup_max_size(); \
  const __acpp_uint64       subgroup_size = __acpp_sscp_get_subgroup_size(); \
  auto local_x = x; \
  for (__acpp_int32 i = lrange / 2; i > 0; i /= 2) {  \
    auto other_x=bit_cast<__acpp_##outType>(__acpp_sscp_sub_group_select_i##size(bit_cast<__acpp_uint##size>(local_x), lid+i)); \
    if (lid +i < subgroup_size) \
        local_x = binary_op(local_x, other_x); \
    } \
    return bit_cast<__acpp_##outType>(__acpp_sscp_sub_group_select_i##size(bit_cast<__acpp_uint##size>(local_x), 0)); \
} \

REDUCE_OVER_GROUP(f32,32)
REDUCE_OVER_GROUP(f64,64)

REDUCE_OVER_GROUP(int8,8)
REDUCE_OVER_GROUP(int16,16)
REDUCE_OVER_GROUP(int32,32)
REDUCE_OVER_GROUP(int64,64)

REDUCE_OVER_GROUP(uint8,8)
REDUCE_OVER_GROUP(uint16,16)
REDUCE_OVER_GROUP(uint32,32)
REDUCE_OVER_GROUP(uint64,64)


#endif

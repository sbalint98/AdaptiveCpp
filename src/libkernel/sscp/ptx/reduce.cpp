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
#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

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

__acpp_uint32 get_active_mask(){
    //__acpp_int64 mask = __nvvm_activemask();
    __acpp_uint32 subgroup_size = __acpp_sscp_get_subgroup_size();
    return (1ull << subgroup_size)-1;
}

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
  const __acpp_uint32 activemask = get_active_mask(); \
  auto local_x = x; \
  for (__acpp_int32 i = lrange / 2; i > 0; i /= 2) {  \
    auto other_x=bit_cast<__acpp_##outType>(__acpp_sscp_sub_group_select_i##size(bit_cast<__acpp_uint##size>(local_x), lid+i)); \
   if (activemask & (1 << (lid + i))) \
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

#define SUBGROUP_INT_REDUCTION(type) \
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

SUBGROUP_INT_REDUCTION(int8)
SUBGROUP_INT_REDUCTION(int16)
SUBGROUP_INT_REDUCTION(int32)
SUBGROUP_INT_REDUCTION(int64)

SUBGROUP_INT_REDUCTION(uint8)
SUBGROUP_INT_REDUCTION(uint16)
SUBGROUP_INT_REDUCTION(uint32)
SUBGROUP_INT_REDUCTION(uint64)
// HIPSYCL_SSCP_CONVERGENT_BUILTIN
// __acpp_f64 __acpp_sscp_sub_group_reduce_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);
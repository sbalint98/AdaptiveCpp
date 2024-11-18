
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


#ifndef HIPSYCL_SSCP_UTILS_BUILTINS_HPP
#define HIPSYCL_SSCP_UTILS_BUILTINS_HPP

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

#endif

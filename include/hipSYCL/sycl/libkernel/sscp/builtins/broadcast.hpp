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
#include "barrier.hpp"
#include "shuffle.hpp"

#ifndef HIPSYCL_SSCP_BROADCAST_BUILTINS_HPP
#define HIPSYCL_SSCP_BROADCAST_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_work_group_broadcast_i8(__acpp_int32 sender,
                                                      __acpp_int8 x);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_work_group_broadcast_i16(__acpp_int32 sender,
                                                        __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_work_group_broadcast_i32(__acpp_int32 sender,
                                                        __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_work_group_broadcast_i64(__acpp_int32 sender,
                                                        __acpp_int64 x);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_broadcast_i8(__acpp_int32 sender,
                                                     __acpp_int8 x);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_broadcast_i16(__acpp_int32 sender,
                                                       __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_broadcast_i32(__acpp_int32 sender,
                                                       __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_broadcast_i64(__acpp_int32 sender,
                                                       __acpp_int64 x);

#define SUBGROUP_BCAST(fn_suffix,input_type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##input_type __acpp_sscp_sub_group_broadcast_##fn_suffix(__acpp_int32 sender, \
                                                     __acpp_##input_type x){ \
    return __acpp_sscp_sub_group_select_##fn_suffix(x, sender); \
                                                     } \

#define GROUP_BCAST(fn_suffix,input_type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN \
__acpp_##input_type __acpp_sscp_work_group_broadcast_##fn_suffix(__acpp_int32 sender, \
                                                     __acpp_##input_type x){ \
     static __attribute__((loader_uninitialized))  __attribute__((address_space(3))) int shrd_x[1]; \
     if(sender == __acpp_sscp_typed_get_local_linear_id<3, int>()){ \
        shrd_x[0] = x; \
     }; \
     __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed); \
     x = shrd_x[0]; \
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed); \
    return x; \
    } \

#endif

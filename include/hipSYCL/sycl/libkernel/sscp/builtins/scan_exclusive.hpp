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

#ifndef HIPSYCL_SSCP_SCAN_EXCLUSIVE_BUILTINS_HPP
#define HIPSYCL_SSCP_SCAN_EXCLUSIVE_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_exclusive_scan_i8(__acpp_sscp_algorithm_op op, __acpp_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_exclusive_scan_i16(__acpp_sscp_algorithm_op op, __acpp_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_exclusive_scan_i32(__acpp_sscp_algorithm_op op, __acpp_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_exclusive_scan_i64(__acpp_sscp_algorithm_op op, __acpp_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint8 __acpp_sscp_sub_group_exclusive_scan_u8(__acpp_sscp_algorithm_op op, __acpp_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint16 __acpp_sscp_sub_group_exclusive_scan_u16(__acpp_sscp_algorithm_op op, __acpp_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint32 __acpp_sscp_sub_group_exclusive_scan_u32(__acpp_sscp_algorithm_op op, __acpp_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_uint64 __acpp_sscp_sub_group_exclusive_scan_u64(__acpp_sscp_algorithm_op op, __acpp_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f16 __acpp_sscp_sub_group_exclusive_scan_f16(__acpp_sscp_algorithm_op op, __acpp_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f32 __acpp_sscp_sub_group_exclusive_scan_f32(__acpp_sscp_algorithm_op op, __acpp_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_f64 __acpp_sscp_sub_group_exclusive_scan_f64(__acpp_sscp_algorithm_op op, __acpp_f64 x);

#endif
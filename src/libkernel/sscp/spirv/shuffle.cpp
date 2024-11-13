
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

template <typename dataT>
dataT __spirv_SubgroupShuffleINTEL(dataT Data, __acpp_uint32 InvocationId) noexcept;
template <typename dataT>
dataT __spirv_SubgroupShuffleDownINTEL(dataT Current, dataT Next,
                                       __acpp_uint32 Delta) noexcept;
template <typename dataT>
dataT __spirv_SubgroupShuffleUpINTEL(dataT Previous, dataT Current,
                                     __acpp_uint32 Delta) noexcept;
template <typename dataT>
dataT __spirv_SubgroupShuffleXorINTEL(dataT Data, __acpp_uint32 Value) noexcept;

template <typename ValueT, typename IdT>
ValueT
    __spirv_GroupNonUniformShuffle(__acpp_uint32, ValueT, IdT) noexcept;

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shl_i8(__acpp_int8 value,
                                         __acpp_uint32 delta)
{
  return __acpp_sscp_sub_group_shl_i32(value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shl_i16(__acpp_int16 value,
                                           __acpp_uint32 delta)
{
  return __acpp_sscp_sub_group_shl_i32(value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shl_i32(__acpp_int32 value,
                                           __acpp_uint32 delta)
{
    __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
    __acpp_int32 target_id = local_id;
    if (local_id >= delta)
      target_id -= delta;
    return __spirv_GroupNonUniformShuffle(3,
                                          value, target_id);
  // return __spirv_SubgroupShuffleDownINTEL(value, value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shl_i64(__acpp_int64 value,
                                           __acpp_uint32 delta)
{
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  __acpp_sscp_sub_group_shl_i32(tmp[0], delta);
  __acpp_sscp_sub_group_shl_i32(tmp[1], delta);
  __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shr_i8(__acpp_int8 value,
                                         __acpp_uint32 delta)
{
  return __acpp_sscp_sub_group_shl_i32(value, delta);
}
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shr_i16(__acpp_int16 value,
                                           __acpp_uint32 delta)
{
  return __acpp_sscp_sub_group_shl_i32(value, delta);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shr_i32(__acpp_int32 value,
                                           __acpp_uint32 delta)
{
    __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
    __acpp_int32 target_id = local_id + delta;
    if (target_id >= __acpp_sscp_get_subgroup_size())
      target_id = local_id;
 return __spirv_GroupNonUniformShuffle(3,
                                          value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shr_i64(__acpp_int64 value,
                                           __acpp_uint32 delta)
{
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  __acpp_sscp_sub_group_shl_i32(tmp[0], delta);
  __acpp_sscp_sub_group_shl_i32(tmp[1], delta);
  __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_permute_i8(__acpp_int8 value,
                                             __acpp_int32 mask)
{
  return __acpp_sscp_sub_group_permute_i32(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_permute_i16(__acpp_int16 value,
                                               __acpp_int32 mask)
{
  return __acpp_sscp_sub_group_permute_i32(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_permute_i32(__acpp_int32 value,
                                               __acpp_int32 mask)
{
  return __spirv_SubgroupShuffleXorINTEL(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_permute_i64(__acpp_int64 value,
                                               __acpp_int32 mask)
{
    __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
    __acpp_int32 target_id = mask ^ local_id;
 return __spirv_GroupNonUniformShuffle(3,
                                          value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_select_i8(__acpp_int8 value,
                                            __acpp_int32 id)
{
  return __acpp_sscp_sub_group_select_i32(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_select_i16(__acpp_int16 value,
                                              __acpp_int32 id)
{
  return __acpp_sscp_sub_group_select_i32(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_select_i32(__acpp_int32 value,
                                              __acpp_int32 id)
{
  // TODO: This should work in theory but it does not unfortunatelly 
  // It was also tried with the WG_ID but it did not make a reference
  // Unfortuantely this way we need the intel extension which is not optimal
  // Reference usage: https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/detail/spirv.hpp#L928-L941
  // first param: https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/__spirv/spirv_types.hpp#L27-L33
  // __acpp_uint32 id_unsigned = id;
  // return __spirv_GroupNonUniformShuffle(3u, value, id_unsigned);
   __acpp_uint32 target_wg_id = __acpp_sscp_get_subgroup_local_id()*__acpp_sscp_get_subgroup_size() + id;
  return __spirv_SubgroupShuffleINTEL(value, target_wg_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_select_i64(__acpp_int64 value,
                                              __acpp_int32 id)
{
  int tmp[2];
  __builtin_memcpy(tmp, &value, sizeof(tmp));
  __acpp_sscp_sub_group_select_i32(tmp[0], id);
  __acpp_sscp_sub_group_select_i32(tmp[1], id);
  __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
  return result;
}

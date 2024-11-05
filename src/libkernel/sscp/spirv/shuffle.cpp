
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


extern "C" int __spirv_BuiltInGroupNonUniformShuffleUp(int, int);
extern "C" int __spirv_BuiltInGroupNonUniformShuffleDown(int, int);
extern "C" int __spirv_BuiltInGroupNonUniformShuffleXor(int, int);
extern "C" int __spirv_BuiltInGroupNonUniformShuffle(int, int);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shl_i8(__acpp_int8 value,
                                                __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shl_i32(value, delta);
                                                }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shl_i16(__acpp_int16 value,
                                                  __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shl_i32(value, delta);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shl_i32(__acpp_int32 value,
                                                  __acpp_uint32 delta){
        
    return __spirv_BuiltInGroupNonUniformShuffleDown(value, delta);
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shl_i64(__acpp_int64 value,
                                                  __acpp_uint32 delta){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    __acpp_sscp_sub_group_shl_i32(tmp[0], delta);
    __acpp_sscp_sub_group_shl_i32(tmp[1], delta);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shr_i8(__acpp_int8 value,
                                                __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shl_i32(value, delta);
                                                }
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shr_i16(__acpp_int16 value,
                                                  __acpp_uint32 delta){
    return __acpp_sscp_sub_group_shl_i32(value, delta);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shr_i32(__acpp_int32 value,
                                                  __acpp_uint32 delta){
    return __spirv_BuiltInGroupNonUniformShuffleUp(value, delta);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shr_i64(__acpp_int64 value,
                                                  __acpp_uint32 delta){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    __acpp_sscp_sub_group_shl_i32(tmp[0], delta);
    __acpp_sscp_sub_group_shl_i32(tmp[1], delta);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_permute_i8(__acpp_int8 value,
                                                   __acpp_int32 mask){
    return __acpp_sscp_sub_group_permute_i32(value, mask);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_permute_i16(__acpp_int16 value,
                                                     __acpp_int32 mask){
    return __acpp_sscp_sub_group_permute_i32(value, mask);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_permute_i32(__acpp_int32 value,
                                                     __acpp_int32 mask){
    return __spirv_BuiltInGroupNonUniformShuffleXor(value, mask);                                                
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_permute_i64(__acpp_int64 value,
                                                     __acpp_int32 mask){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    __acpp_sscp_sub_group_permute_i32(tmp[0], mask);
    __acpp_sscp_sub_group_permute_i32(tmp[1], mask);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_select_i8(__acpp_int8 value,
                                                   __acpp_int32 id){
    return __acpp_sscp_sub_group_select_i32(value, id);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_select_i16(__acpp_int16 value,
                                                     __acpp_int32 id){
    return __acpp_sscp_sub_group_select_i32(value, id);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_select_i32(__acpp_int32 value,
                                                     __acpp_int32 id){
    return __spirv_BuiltInGroupNonUniformShuffle(value, id);
                                                  }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_select_i64(__acpp_int64 value,
                                                     __acpp_int32 id){
    int tmp[2];
    __builtin_memcpy(tmp, &value, sizeof(tmp));
    __acpp_sscp_sub_group_select_i32(tmp[0], id);
    __acpp_sscp_sub_group_select_i32(tmp[1], id);
    __acpp_int64 result = (static_cast<__acpp_int64>(tmp[1]) << 32ull) | (static_cast<__acpp_uint32>(tmp[0]));
    return result;
                                                  }



/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#ifndef HIPSYCL_SSCP_AMDGPU_OCML_INTERFACE_HPP
#define HIPSYCL_SSCP_AMDGPU_OCML_INTERFACE_HPP

#include "../builtin_config.hpp"
#include "ockl.hpp"

struct __hipsycl_int32_array2 {
  __hipsycl_int32 data [2];
};

extern "C" double __ocml_acos_f64 (double);
extern "C" float __ocml_acos_f32 (float);
extern "C" float __ocml_fmuladd_f32 (float, float, float);
extern "C" __hipsycl_native_f16_2 __ocml_acos_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_acos_f16 (__hipsycl_native_f16);
extern "C" double __ocml_acosh_f64 (double);
extern "C" double __ocmlpriv_lnep_f64 (__hipsycl_f64_2, __hipsycl_int32);
extern "C" float __ocml_acosh_f32 (float);
extern "C" float __ocmlpriv_lnep_f32 (__hipsycl_f32_2, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_acosh_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_acosh_f16 (__hipsycl_native_f16);
extern "C" double __ocml_acospi_f64 (double);
extern "C" float __ocml_acospi_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_acospi_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_acospi_f16 (__hipsycl_native_f16);
extern "C" double __ocml_asin_f64 (double);
extern "C" float __ocml_asin_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_asin_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_asin_f16 (__hipsycl_native_f16);
extern "C" double __ocml_asinh_f64 (double);
extern "C" float __ocml_asinh_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_asinh_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_asinh_f16 (__hipsycl_native_f16);
extern "C" double __ocml_asinpi_f64 (double);
extern "C" float __ocml_asinpi_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_asinpi_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_asinpi_f16 (__hipsycl_native_f16);
extern "C" double __ocml_atan2_f64 (double, double);
extern "C" double __ocmlpriv_atanred_f64 (double);
extern "C" float __ocml_atan2_f32 (float, float);
extern "C" float __ocmlpriv_atanred_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_atan2_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_atan2_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocmlpriv_atanred_f16 (__hipsycl_native_f16);
extern "C" double __ocml_atan2pi_f64 (double, double);
extern "C" double __ocmlpriv_atanpired_f64 (double);
extern "C" float __ocml_atan2pi_f32 (float, float);
extern "C" float __ocmlpriv_atanpired_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_atan2pi_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_atan2pi_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocmlpriv_atanpired_f16 (__hipsycl_native_f16);
extern "C" double __ocml_atan_f64 (double);
extern "C" float __ocml_atan_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_atan_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_atan_f16 (__hipsycl_native_f16);
extern "C" double __ocml_atanh_f64 (double);
extern "C" float __ocml_atanh_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_atanh_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_atanh_f16 (__hipsycl_native_f16);
extern "C" double __ocml_atanpi_f64 (double);
extern "C" float __ocml_atanpi_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_atanpi_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_atanpi_f16 (__hipsycl_native_f16);
extern "C" double __ocmlpriv_ba0_f64 (double);
extern "C" float __ocmlpriv_ba0_f32 (float);
extern "C" double __ocmlpriv_ba1_f64 (double);
extern "C" float __ocmlpriv_ba1_f32 (float);
extern "C" double __ocmlpriv_bp0_f64 (double);
extern "C" float __ocmlpriv_bp0_f32 (float);
extern "C" double __ocmlpriv_bp1_f64 (double);
extern "C" float __ocmlpriv_bp1_f32 (float);
extern "C" double __ocml_cabs_f64 (__hipsycl_f64_2);
extern "C" double __ocml_hypot_f64 (double, double);
extern "C" float __ocml_cabs_f32 (__hipsycl_f32_2);
extern "C" float __ocml_hypot_f32 (float, float);
extern "C" __hipsycl_f64_2 __ocml_cacos_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocml_cacosh_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_4 __ocmlpriv_epcsqrtep_f64 (__hipsycl_f64_4);
extern "C" __hipsycl_f32_2 __ocml_cacos_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocml_cacosh_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_4 __ocmlpriv_epcsqrtep_f32 (__hipsycl_f32_4);
extern "C" __hipsycl_f64_2 __ocml_casin_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocml_casinh_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_casin_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocml_casinh_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f64_2 __ocml_catan_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocml_catanh_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_catan_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocml_catanh_f32 (__hipsycl_f32_2);
extern "C" double __ocml_cbrt_f64 (double);
extern "C" float __ocml_cbrt_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_cbrt_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_cbrt_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_f64_2 __ocml_ccos_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocml_ccosh_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocmlpriv_epexpep_f64 (__hipsycl_f64_2);
extern "C" double __ocml_sincos_f64 (double, double);
//extern "C" %0 __ocmlpriv_trigred_f64 (double);
//extern "C" %1 __ocmlpriv_sincosred2_f64 (double, double);
//extern "C" %0 __ocmlpriv_trigredsmall_f64 (double);
//extern "C" %0 __ocmlpriv_trigredlarge_f64 (double);
extern "C" __hipsycl_f32_2 __ocml_ccos_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocml_ccosh_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocmlpriv_epexpep_f32 (__hipsycl_f32_2);
extern "C" float __ocml_sincos_f32 (float, float);
extern "C" __hipsycl_int32_array2 __ocmlpriv_trigred_f32 (float);
extern "C" __hipsycl_int32_array2 __ocmlpriv_sincosred_f32 (float);
extern "C" __hipsycl_int32_array2 __ocmlpriv_trigredsmall_f32 (float);
extern "C" __hipsycl_int32_array2 __ocmlpriv_trigredlarge_f32 (float);
extern "C" __hipsycl_f64_2 __ocml_cdiv_f64 (__hipsycl_f64_2, __hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_cdiv_f32 (__hipsycl_f32_2, __hipsycl_f32_2);
extern "C" double __ocml_ceil_f64 (double);
extern "C" float __ocml_ceil_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_ceil_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_ceil_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_f64_2 __ocml_cexp_f64 (__hipsycl_f64_2);
extern "C" double __ocml_exp_f64 (double);
extern "C" __hipsycl_f32_2 __ocml_cexp_f32 (__hipsycl_f32_2);
extern "C" float __ocml_exp_f32 (float);
extern "C" __hipsycl_f64_2 __ocml_clog_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_clog_f32 (__hipsycl_f32_2);
extern "C" double __ocml_copysign_f64 (double, double);
extern "C" float __ocml_copysign_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_copysign_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_copysign_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_cos_f64 (double);
extern "C" float __ocml_cos_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_cos_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_cos_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocmlpriv_trigred_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocmlpriv_sincosred_f16 (__hipsycl_native_f16);
extern "C" double __ocmlpriv_cosb_f64 (double, __hipsycl_int32, double);
extern "C" float __ocmlpriv_cosb_f32 (float, __hipsycl_int32, float);
extern "C" double __ocml_cosh_f64 (double);
extern "C" float __ocml_cosh_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_cosh_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_cosh_f16 (__hipsycl_native_f16);
extern "C" double __ocml_cospi_f64 (double);
extern "C" float __ocml_cospi_f32 (float);
extern "C" __hipsycl_int32_array2 __ocmlpriv_trigpired_f32 (float);
extern "C" __hipsycl_int32_array2 __ocmlpriv_sincospired_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_cospi_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_cospi_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocmlpriv_trigpired_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocmlpriv_sincospired_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_f64_2 __ocml_csin_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocml_csinh_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_csin_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocml_csinh_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f64_2 __ocml_csqrt_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_csqrt_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f64_2 __ocml_ctan_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f64_2 __ocml_ctanh_f64 (__hipsycl_f64_2);
extern "C" __hipsycl_f32_2 __ocml_ctan_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f32_2 __ocml_ctanh_f32 (__hipsycl_f32_2);
extern "C" __hipsycl_f64_2 __ocmlpriv_epln_f64 (double);
extern "C" __hipsycl_f32_2 __ocmlpriv_epln_f32 (float);
extern "C" double __ocml_erf_f64 (double);
extern "C" float __ocml_erf_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_erf_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_erf_f16 (__hipsycl_native_f16);
extern "C" double __ocml_erfc_f64 (double);
extern "C" double __ocmlpriv_erfcx_f64 (double);
extern "C" float __ocml_erfc_f32 (float);
extern "C" float __ocmlpriv_erfcx_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_erfc_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_erfc_f16 (__hipsycl_native_f16);
extern "C" double __ocml_erfcinv_f64 (double);
extern "C" double __ocml_erfinv_f64 (double);
extern "C" double __ocml_log_f64 (double);
extern "C" float __ocml_erfcinv_f32 (float);
extern "C" float __ocml_erfinv_f32 (float);
extern "C" float __ocml_log_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_erfcinv_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_erfcinv_f16 (__hipsycl_native_f16);
extern "C" double __ocml_erfcx_f64 (double);
extern "C" float __ocml_erfcx_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_erfcx_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_erfcx_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16_2 __ocml_erfinv_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_erfinv_f16 (__hipsycl_native_f16);
extern "C" double __ocml_exp10_f64 (double);
extern "C" float __ocml_exp10_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_exp10_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_exp10_f16 (__hipsycl_native_f16);
extern "C" double __ocml_exp2_f64 (double);
extern "C" float __ocml_exp2_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_exp2_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_exp2_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16_2 __ocml_exp_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_exp_f16 (__hipsycl_native_f16);
extern "C" double __ocmlpriv_expep_f64 (__hipsycl_f64_2);
extern "C" float __ocmlpriv_expep_f32 (__hipsycl_f32_2);
extern "C" double __ocml_expm1_f64 (double);
extern "C" float __ocml_expm1_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_expm1_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_expm1_f16 (__hipsycl_native_f16);
extern "C" double __ocml_fabs_f64 (double);
extern "C" float __ocml_fabs_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_fabs_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fabs_f16 (__hipsycl_native_f16);
extern "C" double __ocml_fdim_f64 (double, double);
extern "C" float __ocml_fdim_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_fdim_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fdim_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_floor_f64 (double);
extern "C" float __ocml_floor_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_floor_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_floor_f16 (__hipsycl_native_f16);
extern "C" double __ocml_fma_f64 (double, double, double);
extern "C" __hipsycl_f32_2 __ocml_fma_2f32 (__hipsycl_f32_2, __hipsycl_f32_2, __hipsycl_f32_2);
extern "C" float __ocml_fma_f32 (float, float, float);
extern "C" __hipsycl_native_f16_2 __ocml_fma_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fma_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_fmax_f64 (double, double);
extern "C" float __ocml_fmax_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_fmax_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fmax_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_fmin_f64 (double, double);
extern "C" float __ocml_fmin_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_fmin_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fmin_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_fmod_f64 (double, double);
extern "C" float __ocml_fmod_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_fmod_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fmod_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_fmuladd_f64 (double, double, double);
extern "C" __hipsycl_f32_2 __ocml_fmuladd_2f32 (__hipsycl_f32_2, __hipsycl_f32_2, __hipsycl_f32_2);
extern "C" __hipsycl_native_f16_2 __ocml_fmuladd_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_fmuladd_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocml_fpclassify_f64 (double);
extern "C" __hipsycl_int32 __ocml_fpclassify_f32 (float);
extern "C" __hipsycl_int32 __ocml_fpclassify_f16 (__hipsycl_native_f16);
extern "C" double __ocml_fract_f64 (double, __amdgpu_private double*);
extern "C" float __ocml_fract_f32 (float, __amdgpu_private float*);
extern "C" __hipsycl_native_f16_2 __ocml_fract_2f16 (__hipsycl_native_f16_2, __amdgpu_private __hipsycl_native_f16_2*);
extern "C" __hipsycl_native_f16 __ocml_fract_f16 (__hipsycl_native_f16, __amdgpu_private __hipsycl_native_f16*);
extern "C" double __ocml_frexp_f64 (double,  __amdgpu_private __hipsycl_int32 *);
extern "C" float __ocml_frexp_f32 (float,  __amdgpu_private __hipsycl_int32*);
extern "C" __hipsycl_native_f16_2 __ocml_frexp_2f16 (__hipsycl_native_f16_2,  __amdgpu_private __hipsycl_int32_2*);
extern "C" __hipsycl_native_f16 __ocml_frexp_f16 (__hipsycl_native_f16,  __amdgpu_private __hipsycl_int32*);
extern "C" __hipsycl_native_f16_2 __ocml_hypot_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_hypot_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_i0_f64 (double);
extern "C" double __ocml_rsqrt_f64 (double);
extern "C" float __ocml_i0_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_i0_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_i0_f16 (__hipsycl_native_f16);
extern "C" double __ocml_i1_f64 (double);
extern "C" float __ocml_i1_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_i1_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_i1_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocml_ilogb_f64 (double);
extern "C" __hipsycl_int32 __ocml_ilogb_f32 (float);
extern "C" __hipsycl_int32_2 __ocml_ilogb_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_int32 __ocml_ilogb_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocml_isfinite_f64 (double);
extern "C" __hipsycl_int32 __ocml_isfinite_f32 (float);
extern "C" __hipsycl_int16_2 __ocml_isfinite_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_int32 __ocml_isfinite_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocml_isinf_f64 (double);
extern "C" __hipsycl_int32 __ocml_isinf_f32 (float);
extern "C" __hipsycl_int16_2 __ocml_isinf_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_int32 __ocml_isinf_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocml_isnan_f64 (double);
extern "C" __hipsycl_int32 __ocml_isnan_f32 (float);
extern "C" __hipsycl_int16_2 __ocml_isnan_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_int32 __ocml_isnan_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_int32 __ocml_isnormal_f64 (double);
extern "C" __hipsycl_int32 __ocml_isnormal_f32 (float);
extern "C" __hipsycl_int16_2 __ocml_isnormal_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_int32 __ocml_isnormal_f16 (__hipsycl_native_f16);
extern "C" double __ocml_j0_f64 (double);
extern "C" float __ocml_j0_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_j0_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_j0_f16 (__hipsycl_native_f16);
extern "C" double __ocml_j1_f64 (double);
extern "C" float __ocml_j1_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_j1_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_j1_f16 (__hipsycl_native_f16);
extern "C" double __ocml_ldexp_f64 (double, __hipsycl_int32);
extern "C" float __ocml_ldexp_f32 (float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_ldexp_2f16 (__hipsycl_native_f16_2, __hipsycl_int32_2);
extern "C" __hipsycl_native_f16 __ocml_ldexp_f16 (__hipsycl_native_f16, __hipsycl_int32);
extern "C" double __ocml_len3_f64 (double, double, double);
extern "C" float __ocml_len3_f32 (float, float, float);
extern "C" __hipsycl_native_f16 __ocml_len3_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_len4_f64 (double, double, double, double);
extern "C" float __ocml_len4_f32 (float, float, float, float);
extern "C" __hipsycl_native_f16 __ocml_len4_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_lgamma_f64 (double);
extern "C" double __ocml_lgamma_r_f64 (double, __amdgpu_private __hipsycl_int32*);
extern "C" double __ocml_sinpi_f64 (double);
extern "C" float __ocml_lgamma_f32 (float);
extern "C" float __ocml_lgamma_r_f32 (float,  __amdgpu_private __hipsycl_int32*);
extern "C" float __ocml_sinpi_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_lgamma_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_lgamma_r_f16 (__hipsycl_native_f16,  __amdgpu_private __hipsycl_int32*);
extern "C" __hipsycl_native_f16 __ocml_lgamma_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16_2 __ocml_lgamma_r_2f16 (__hipsycl_native_f16_2, __amdgpu_private __hipsycl_int32_2*);
extern "C" double __ocml_log10_f64 (double);
extern "C" float __ocml_log10_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_log10_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_log10_f16 (__hipsycl_native_f16);
extern "C" double __ocml_log1p_f64 (double);
extern "C" float __ocml_log1p_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_log1p_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_log1p_f16 (__hipsycl_native_f16);
extern "C" double __ocml_log2_f64 (double);
extern "C" float __ocml_log2_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_log2_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_log2_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16_2 __ocml_log_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_log_f16 (__hipsycl_native_f16);
extern "C" double __ocml_logb_f64 (double);
extern "C" float __ocml_logb_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_logb_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_logb_f16 (__hipsycl_native_f16);
extern "C" double __ocml_mad_f64 (double, double, double);
extern "C" __hipsycl_f32_2 __ocml_mad_2f32 (__hipsycl_f32_2, __hipsycl_f32_2, __hipsycl_f32_2);
extern "C" float __ocml_mad_f32 (float, float, float);
extern "C" __hipsycl_native_f16_2 __ocml_mad_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_mad_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_max_f64 (double, double);
extern "C" float __ocml_max_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_max_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_max_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_maxmag_f64 (double, double);
extern "C" float __ocml_maxmag_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_maxmag_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_maxmag_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_min_f64 (double, double);
extern "C" float __ocml_min_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_min_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_min_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_minmag_f64 (double, double);
extern "C" float __ocml_minmag_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_minmag_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_minmag_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_modf_f64 (double, __amdgpu_private double*);
extern "C" float __ocml_modf_f32 (float, __amdgpu_private float*);
extern "C" __hipsycl_native_f16_2 __ocml_modf_2f16 (__hipsycl_native_f16_2, __amdgpu_private __hipsycl_native_f16_2*);
extern "C" __hipsycl_native_f16 __ocml_modf_f16 (__hipsycl_native_f16, __amdgpu_private __hipsycl_native_f16*);
extern "C" double __ocml_nan_f64 (__hipsycl_int64);
extern "C" float __ocml_nan_f32 (__hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_nan_2f16 (__hipsycl_int16_2);
extern "C" __hipsycl_native_f16 __ocml_nan_f16 (__hipsycl_int16);
extern "C" double __ocml_native_recip_f64 (double);
extern "C" double __ocml_native_sqrt_f64 (double);
extern "C" double __ocml_native_rsqrt_f64 (double);
extern "C" double __ocml_native_sin_f64 (double);
extern "C" double __ocml_native_cos_f64 (double);
extern "C" double __ocml_native_exp_f64 (double);
extern "C" double __ocml_native_exp2_f64 (double);
extern "C" double __ocml_native_log_f64 (double);
extern "C" double __ocml_native_log2_f64 (double);
extern "C" double __ocml_native_log10_f64 (double);
extern "C" float __ocml_native_recip_f32 (float);
extern "C" float __ocml_native_sqrt_f32 (float);
extern "C" float __ocml_native_rsqrt_f32 (float);
extern "C" float __ocml_native_sin_f32 (float);
extern "C" float __ocml_native_cos_f32 (float);
extern "C" float __ocml_native_exp_f32 (float);
extern "C" float __ocml_native_exp2_f32 (float);
extern "C" float __ocml_native_exp10_f32 (float);
extern "C" float __ocml_native_log_f32 (float);
extern "C" float __ocml_native_log2_f32 (float);
extern "C" float __ocml_native_log10_f32 (float);
extern "C" __hipsycl_native_f16 __ocml_native_rcp_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_sqrt_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_rsqrt_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_sin_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_cos_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_exp_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_exp2_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_log_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_log2_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocml_native_log10_f16 (__hipsycl_native_f16);
extern "C" double __ocml_ncdf_f64 (double);
extern "C" float __ocml_ncdf_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_ncdf_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_ncdf_f16 (__hipsycl_native_f16);
extern "C" double __ocml_ncdfinv_f64 (double);
extern "C" float __ocml_ncdfinv_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_ncdfinv_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_ncdfinv_f16 (__hipsycl_native_f16);
extern "C" double __ocml_nearbyint_f64 (double);
extern "C" float __ocml_nearbyint_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_nearbyint_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_nearbyint_f16 (__hipsycl_native_f16);
extern "C" double __ocml_nextafter_f64 (double, double);
extern "C" float __ocml_nextafter_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_nextafter_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_nextafter_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_pow_f64 (double, double);
extern "C" float __ocml_pow_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_pow_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_pow_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_pown_f64 (double, __hipsycl_int32);
extern "C" float __ocml_pown_f32 (float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_pown_2f16 (__hipsycl_native_f16_2, __hipsycl_int32_2);
extern "C" __hipsycl_native_f16 __ocml_pown_f16 (__hipsycl_native_f16, __hipsycl_int32);
extern "C" double __ocml_powr_f64 (double, double);
extern "C" float __ocml_powr_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_powr_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_powr_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_rcbrt_f64 (double);
extern "C" float __ocml_rcbrt_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_rcbrt_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_rcbrt_f16 (__hipsycl_native_f16);
extern "C" double __ocml_remainder_f64 (double, double);
extern "C" float __ocml_remainder_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_remainder_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_remainder_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_remquo_f64 (double, double, __hipsycl_int32);
extern "C" float __ocml_remquo_f32 (float, float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_remquo_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2, __hipsycl_int32_2);
extern "C" __hipsycl_native_f16 __ocml_remquo_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_int32);
extern "C" double __ocml_rhypot_f64 (double, double);
extern "C" float __ocml_rhypot_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_rhypot_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_rhypot_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_rint_f64 (double);
extern "C" float __ocml_rint_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_rint_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_rint_f16 (__hipsycl_native_f16);
extern "C" double __ocml_rlen3_f64 (double, double, double);
extern "C" float __ocml_rlen3_f32 (float, float, float);
extern "C" __hipsycl_native_f16 __ocml_rlen3_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_rlen4_f64 (double, double, double, double);
extern "C" float __ocml_rlen4_f32 (float, float, float, float);
extern "C" __hipsycl_native_f16 __ocml_rlen4_f16 (__hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_rootn_f64 (double, __hipsycl_int32);
extern "C" float __ocml_rootn_f32 (float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_rootn_2f16 (__hipsycl_native_f16_2, __hipsycl_int32_2);
extern "C" __hipsycl_native_f16 __ocml_rootn_f16 (__hipsycl_native_f16, __hipsycl_int32);
extern "C" double __ocml_round_f64 (double);
extern "C" float __ocml_round_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_round_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_round_f16 (__hipsycl_native_f16);
extern "C" float __ocml_rsqrt_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_rsqrt_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_rsqrt_f16 (__hipsycl_native_f16);
extern "C" double __ocml_scalb_f64 (double, double);
extern "C" float __ocml_scalb_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_scalb_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_scalb_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_scalbn_f64 (double, __hipsycl_int32);
extern "C" float __ocml_scalbn_f32 (float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_scalbn_2f16 (__hipsycl_native_f16_2, __hipsycl_int32_2);
extern "C" __hipsycl_native_f16 __ocml_scalbn_f16 (__hipsycl_native_f16, __hipsycl_int32);
extern "C" __hipsycl_int32 __ocml_signbit_f64 (double);
extern "C" __hipsycl_int32 __ocml_signbit_f32 (float);
extern "C" __hipsycl_int16_2 __ocml_signbit_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_int32 __ocml_signbit_f16 (__hipsycl_native_f16);
extern "C" double __ocml_sin_f64 (double);
extern "C" float __ocml_sin_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_sin_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_sin_f16 (__hipsycl_native_f16);
extern "C" double __ocmlpriv_sinb_f64 (double, __hipsycl_int32, double);
extern "C" float __ocmlpriv_sinb_f32 (float, __hipsycl_int32, float);
extern "C" __hipsycl_native_f16_2 __ocml_sincos_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_sincos_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" double __ocml_sincospi_f64 (double, double);
extern "C" float __ocml_sincospi_f32 (float, float);
extern "C" __hipsycl_native_f16_2 __ocml_sincospi_2f16 (__hipsycl_native_f16_2, __hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_sincospi_f16 (__hipsycl_native_f16, __hipsycl_native_f16);
extern "C" __hipsycl_int32_array2 __ocmlpriv_sincosred2_f32 (float, float);
extern "C" double __ocml_sinh_f64 (double);
extern "C" float __ocml_sinh_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_sinh_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_sinh_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16_2 __ocml_sinpi_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_sinpi_f16 (__hipsycl_native_f16);
extern "C" double __ocml_sqrt_f64 (double);
extern "C" float __ocml_sqrt_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_sqrt_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_sqrt_f16 (__hipsycl_native_f16);
extern "C" double __ocml_tan_f64 (double);
extern "C" double __ocmlpriv_tanred2_f64 (double, double, __hipsycl_int32);
extern "C" float __ocml_tan_f32 (float);
extern "C" float __ocmlpriv_tanred_f32 (float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_tan_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_tan_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocmlpriv_tanred_f16 (__hipsycl_native_f16, __hipsycl_int16);
extern "C" double __ocml_tanh_f64 (double);
extern "C" float __ocml_tanh_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_tanh_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_tanh_f16 (__hipsycl_native_f16);
extern "C" double __ocml_tanpi_f64 (double);
extern "C" double __ocmlpriv_tanpired_f64 (double, __hipsycl_int32);
extern "C" float __ocml_tanpi_f32 (float);
extern "C" float __ocmlpriv_tanpired_f32 (float, __hipsycl_int32);
extern "C" __hipsycl_native_f16_2 __ocml_tanpi_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_tanpi_f16 (__hipsycl_native_f16);
extern "C" __hipsycl_native_f16 __ocmlpriv_tanpired_f16 (__hipsycl_native_f16, __hipsycl_int16);
extern "C" double __ocml_tgamma_f64 (double);
extern "C" float __ocml_tgamma_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_tgamma_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_tgamma_f16 (__hipsycl_native_f16);
extern "C" double __ocml_trunc_f64 (double);
extern "C" float __ocml_trunc_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_trunc_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_trunc_f16 (__hipsycl_native_f16);
extern "C" double __ocml_y0_f64 (double);
extern "C" float __ocml_y0_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_y0_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_y0_f16 (__hipsycl_native_f16);
extern "C" double __ocml_y1_f64 (double);
extern "C" float __ocml_y1_f32 (float);
extern "C" __hipsycl_native_f16_2 __ocml_y1_2f16 (__hipsycl_native_f16_2);
extern "C" __hipsycl_native_f16 __ocml_y1_f16 (__hipsycl_native_f16);

#endif

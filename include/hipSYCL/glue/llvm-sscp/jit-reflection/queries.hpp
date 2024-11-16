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
#ifndef ACPP_GLUE_JIT_REFLECTION_QUERIES_HPP
#define ACPP_GLUE_JIT_REFLECTION_QUERIES_HPP


namespace hipsycl{
namespace sycl {
namespace jit {

enum class compiler_backend : int {
  spirv = 0,
  ptx = 1,
  amdgpu = 2,
  host = 3
};

namespace vendor_id {

inline constexpr int nvidia = 4318;
inline constexpr int amd = 1022;
inline constexpr int intel = 8086;

}

}
}
}



extern "C" bool __acpp_sscp_jit_reflect_knows_target_vendor_id();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_arch();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_has_independent_forward_progress();
extern "C" bool __acpp_sscp_jit_reflect_knows_runtime_backend();
extern "C" bool __acpp_sscp_jit_reflect_knows_compiler_backend();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_is_cpu();
extern "C" bool __acpp_sscp_jit_reflect_knows_target_is_gpu();

extern "C" int __acpp_sscp_jit_reflect_target_vendor_id();
extern "C" int __acpp_sscp_jit_reflect_target_arch();
extern "C" bool __acpp_sscp_jit_reflect_target_is_cpu();
extern "C" bool __acpp_sscp_jit_reflect_target_is_gpu();
extern "C" bool __acpp_sscp_jit_reflect_target_has_independent_forward_progress();
extern "C" int __acpp_sscp_jit_reflect_runtime_backend();
extern "C" hipsycl::sycl::jit::compiler_backend __acpp_sscp_jit_reflect_compiler_backend();


#endif

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

#include "backend.hpp"
#include "memory.hpp"
#include "detail/builtin_dispatch.hpp"
#include "hipSYCL/sycl/jit.hpp"


#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "sscp/builtins/amdgpu_dpp.hpp"
#endif
namespace jit = sycl::AdaptiveCpp_jit;



int noop(){
    return 42;
};

namespace adaptiveCpp::sscp::amd_builtins {

namespace{

    template <int dpp_ctrl_code>
    int __internal_acpp_sscp_dpp_builtin(int value) {
        if(jit::reflect<jit::reflection_query::compiler_backend>() ==
              jit::compiler_backend::amdgpu){
                return __acpp_sscp_dpp_builtin<dpp_ctrl_code>(value);
              }
        else {
            return 191919191;
        }
    }
}

template <int dpp_ctrl_code>
HIPSYCL_BUILTIN int acpp_sscp_dpp_builtin(int value) noexcept {
    __acpp_backend_switch(return noop(), return __internal_acpp_sscp_dpp_builtin<dpp_ctrl_code>(value) , return noop(), return noop());
}


}





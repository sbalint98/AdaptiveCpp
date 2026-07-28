// REQUIRES: ptx-backend-tools
// RUN: %acpp %s -c -o %t.o -mllvm -acpp-sscp-emit-hcf
// RUN: %llvm-to-ptx --ir %s.hcf %t.bc llvm-ir.global
// RUN: rm -f %s.hcf
// RUN: %llvm-dis %t.bc -o %t.ll
// RUN: FileCheck --check-prefix=CHECK-IR %s < %t.ll
// RUN: llc -march=nvptx64 -mcpu=sm_80 %t.ll -o %t.s
// RUN: FileCheck --check-prefix=CHECK-ASM %s < %t.s

#include <sycl/sycl.hpp>
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx_auto_builtins.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"

namespace jit = hipsycl::sycl::AdaptiveCpp_jit;

// CHECK-IR: define ptx_kernel void @{{.*}}basic_parallel_for{{.*}}

void test_builtins(unsigned int *dev_activemask, int global_idx) {
  __acpp_if_target_sscp(
    jit::compile_if(
      jit::reflect<jit::reflection_query::compiler_backend>() ==
        jit::compiler_backend::ptx,
      [&]() {
        // ── activemask ──────────────────────────────────────────────────────
        // CHECK-IR: call{{.*}} i32 @llvm.nvvm.activemask()
        // CHECK-ASM: activemask.b32
        *dev_activemask = __acpp___nvvm_activemask();

        // ── abs_bf16 ────────────────────────────────────────────────────────
        // CHECK-IR: call{{.*}} bfloat @llvm.nvvm.fabs.bf16
        // CHECK-ASM: abs.bf16
        __bf16 bf16_val = 0;
        __bf16 bf16_abs = __acpp___nvvm_abs_bf16(bf16_val);
        // store to avoid optimization
        *reinterpret_cast<__bf16*>(dev_activemask) = bf16_abs;
      }
    );
  );
}

int main() {
  sycl::queue q;

  unsigned int *dev_activemask = sycl::malloc_device<unsigned int>(1024, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> idx) {
      test_builtins(dev_activemask, idx[0]);
    });
  }).wait();

  sycl::free(dev_activemask, q);
}

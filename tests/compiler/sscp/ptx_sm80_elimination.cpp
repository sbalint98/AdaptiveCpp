// REQUIRES: ptx-backend-tools
// RUN: %acpp %s -c -o %t.o -mllvm -acpp-sscp-emit-hcf

// Test 1: Compile for sm_80. The compile_if for sm_80 should be kept.
// RUN: %llvm-to-ptx --ir --build-opt ptx-target-device=80 %s.hcf %t.sm80.bc llvm-ir.global
// RUN: %llvm-dis %t.sm80.bc -o %t.sm80.ll
// RUN: FileCheck --check-prefix=CHECK-SM80-IR %s < %t.sm80.ll
// RUN: llc -march=nvptx64 -mcpu=sm_80 %t.sm80.ll -o %t.sm80.s
// RUN: FileCheck --check-prefix=CHECK-SM80-ASM %s < %t.sm80.s

// Test 2: Compile for sm_70. The compile_if for sm_80 should be eliminated!
// RUN: %llvm-to-ptx --ir --build-opt ptx-target-device=70 %s.hcf %t.sm70.bc llvm-ir.global
// RUN: %llvm-dis %t.sm70.bc -o %t.sm70.ll
// RUN: FileCheck --check-prefix=CHECK-SM70-IR %s < %t.sm70.ll
// RUN: llc -march=nvptx64 -mcpu=sm_70 %t.sm70.ll -o %t.sm70.s
// RUN: FileCheck --check-prefix=CHECK-SM70-ASM %s < %t.sm70.s

#include <sycl/sycl.hpp>
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx_auto_builtins.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"

namespace jit = hipsycl::sycl::AdaptiveCpp_jit;

void test_builtins(int *out) {
  __acpp_if_target_sscp(
    jit::compile_if(
      jit::reflect<jit::reflection_query::compiler_backend>() ==
        jit::compiler_backend::ptx,
      [&]() {
        // Redux sync is only available on sm_80+
        jit::compile_if_else(
          jit::reflect<jit::reflection_query::target_arch>() >= 80,
          [&]() {
            // CHECK-SM80-IR: call i32 @llvm.nvvm.redux.sync.add
            // CHECK-SM80-ASM: redux.sync.add.s32
            // CHECK-SM70-IR-NOT: call i32 @llvm.nvvm.redux.sync.add
            // CHECK-SM70-ASM-NOT: redux.sync.add
            *out = __acpp___nvvm_redux_sync_add(1, 0xFFFFFFFF);
          },
          [&]() {
            // Fallback for older architectures
            *out = 0;
          }
        );
      }
    );
  );
}

int main() {
  sycl::queue q;
  int *dev_out = sycl::malloc_device<int>(1, q);
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      test_builtins(dev_out);
    });
  }).wait();
  sycl::free(dev_out, q);
}

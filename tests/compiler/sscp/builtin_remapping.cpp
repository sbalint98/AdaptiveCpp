
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s --implicit-check-not='{{^0$}}'
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s --implicit-check-not='{{^0$}}'
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math
// RUN: %t | FileCheck %s --implicit-check-not='{{^0$}}'
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s --implicit-check-not='{{^0$}}'

#include <iostream>
#include <cmath>
#include <limits>
#include <sycl/sycl.hpp>
#include "common.hpp"

#ifdef __FAST_MATH__
constexpr bool is_fast_math = true;
#else
constexpr bool is_fast_math = false;
#endif

#pragma float_control(precise, on, push)

template<class T>
bool check_with_tolerance(T actual, double expected) {
  constexpr double tol =
      is_fast_math ? 5e-3
                   : 32.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
  double a = static_cast<double>(actual);
  return std::abs(a - expected) <= tol * std::max(std::abs(a), std::abs(expected));
}

template<class T>
void verify_results(const std::vector<T>& host, T init) {
  double init_d = static_cast<double>(init);

  // CHECK: 1
  std::cout << check_with_tolerance(host[0], std::sin(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[1], std::cos(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[2], std::pow(init_d, init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[3], std::pow(init_d, 3.0)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[4], std::exp(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[5], std::sqrt(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[6], std::tan(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[7], std::exp2(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[8], std::log(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[9], std::asin(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[10], std::fdim(init_d, 0.5)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[11], std::erf(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[12], std::erfc(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[13], std::atan2(init_d, 0.5)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[14], std::fabs(-init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[15], std::floor(init_d)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[16], std::fmax(init_d, 0.5)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[17], std::fmin(init_d, 0.5)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[18], std::fmod(init_d, 0.5)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[19], std::fma(init_d, 0.5, 0.25)) << std::endl;
}

#pragma float_control(pop)


template<class T>
void test() {
  sycl::queue q = get_queue();

  int num_functions = 20;

  T init = static_cast<T>(0.75);

  T* data = sycl::malloc_device<T>(num_functions, q);
  for(int i = 0; i < num_functions; ++i)
    q.memcpy(data + i, &init, sizeof(T));
  q.wait();

  q.single_task([=](){
    data[0] = std::sin(data[0]);
    data[1] = std::cos(data[1]);
    data[2] = std::pow(data[2], init);
    // pow(float, int) is missing in standard after C++11,
    // will be replaced to llvm.pow.f32.i32 for -O3 -ffast-math
    data[3] = std::pow(data[3], T(3));
    data[4] = std::exp(data[4]);
    data[5] = std::sqrt(data[5]);
    data[6] = std::tan(data[6]);
    data[7] = std::exp2(data[7]);
    data[8] = std::log(data[8]);
    data[9] = std::asin(data[9]);
    data[10] = std::fdim(data[10], T(0.5));
    data[11] = std::erf(data[11]);
    data[12] = std::erfc(data[12]);
    data[13] = std::atan2(data[13], T(0.5));
    data[14] = std::fabs(-data[14]);
    data[15] = std::floor(data[15]);
    data[16] = std::fmax(data[16], T(0.5));
    data[17] = std::fmin(data[17], T(0.5));
    data[18] = std::fmod(data[18], T(0.5));
    data[19] = std::fma(data[19], T(0.5), T(0.25));
  }).wait();

  std::vector<T> host(num_functions, T(0));
  q.memcpy(host.data(), data, sizeof(T) * num_functions).wait();

  verify_results(host, init);

  sycl::free(data, q);
}

int main() {
  test<float>();
  if(get_queue().get_device().has(sycl::aspect::fp64))
    test<double>();
}

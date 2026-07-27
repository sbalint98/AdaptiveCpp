add_executable(acpp-amdgpu-builtin-gen main.cpp)

configure_target(TARGET acpp-amdgpu-builtin-gen)

if(NOT WIN32 AND NOT ACPP_LLVM_COMPONENT)
  # On Unix, we need to explicitly link the clang-cpp and LLVM shared libraries
  # since configure_target is tailored for plugins and doesn't link them automatically.
  find_library(CLANG_CPP_LIB NAMES clang-cpp PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
  find_library(LLVM_LIB NAMES LLVM PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)

  if (NOT CLANG_CPP_LIB)
    set(CLANG_CPP_LIB clang-cpp)
  endif()
  if (NOT LLVM_LIB)
    set(LLVM_LIB LLVM)
  endif()

  target_link_options(acpp-amdgpu-builtin-gen PRIVATE "-Wl,-rpath,${LLVM_LIBRARY_DIR}")

  target_link_libraries(acpp-amdgpu-builtin-gen PRIVATE
    ${CLANG_CPP_LIB}
    ${LLVM_LIB}
  )
endif()

if(ACPP_LLVM_COMPONENT)
  # When building in-tree with LLVM, we need to explicitly include the Clang headers
  # because LLVM_INCLUDE_DIRS only covers LLVM, not Clang.
  target_include_directories(acpp-amdgpu-builtin-gen PRIVATE
    ${LLVM_MAIN_SRC_DIR}/../clang/include
    ${LLVM_BINARY_DIR}/tools/clang/include
  )
endif()

# Optional: ensure we compile with C++17 or whatever AdaptiveCpp uses
set_property(TARGET acpp-amdgpu-builtin-gen PROPERTY CXX_STANDARD 17)


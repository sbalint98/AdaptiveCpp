
if(WITH_LLVM_TO_PTX)
  if(WIN32)
    set(EXTRA_FLAGS -fshort-wchar)
  endif()

  set(GENERATED_HPP "${HIPSYCL_SOURCE_DIR}/include/hipSYCL/sycl/libkernel/sscp/builtins/ptx_auto_builtins.hpp")
  set(GENERATED_CPP "${CMAKE_CURRENT_SOURCE_DIR}/ptx_auto_builtins.cpp")

  add_custom_command(
      OUTPUT ${GENERATED_HPP} ${GENERATED_CPP}
      COMMAND acpp-builtin-gen ptx ${GENERATED_HPP} ${GENERATED_CPP}
      DEPENDS acpp-builtin-gen
      COMMENT "Auto-generating PTX builtin wrappers via LibTooling..."
  )

  add_custom_target(generate_ptx_builtins ALL DEPENDS ${GENERATED_HPP} ${GENERATED_CPP})

  set_source_files_properties(${GENERATED_CPP} PROPERTIES GENERATED TRUE)

  libkernel_generate_bitcode_target(
      TARGETNAME ptx 
      TRIPLE nvptx64-nvidia-cuda
      SOURCES 
      ptx_auto_builtins.cpp
      assert.cpp
      atomic.cpp
      barrier.cpp
      core.cpp
      half.cpp
      integer.cpp
      print.cpp
      relational.cpp
      math.cpp
      native.cpp
      localmem.cpp
      subgroup.cpp
      shuffle.cpp
      reduction.cpp
      broadcast.cpp
      scan_inclusive.cpp
      scan_exclusive.cpp
      collpredicate.cpp
      ADDITIONAL_ARGS -Xclang -target-feature -Xclang +sm_60 -Xclang -target-feature -Xclang +ptx60 ${EXTRA_FLAGS})
endif()

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include <iostream>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;



class BuiltinGenConsumer : public ASTConsumer {
  std::string Arch;
  std::string HppPath;
  std::string CppPath;
public:
  BuiltinGenConsumer(std::string Arch, std::string Hpp, std::string Cpp) : Arch(Arch), HppPath(Hpp), CppPath(Cpp) {}

  std::string sanitizeType(std::string T) {
    size_t pos;
    while ((pos = T.find("_Bool")) != std::string::npos) {
      T.replace(pos, 5, "bool");
    }
    while ((pos = T.find(" __attribute__((ext_vector_type(")) != std::string::npos) {
      size_t endPos = T.find(")))", pos);
      if (endPos != std::string::npos) {
        std::string num = T.substr(pos + 32, endPos - (pos + 32));
        std::string base = T.substr(0, pos);
        
        std::string suffix = "";
        while (base.length() > 0 && (base.back() == '*' || base.back() == ' ')) {
            suffix = base.back() + suffix;
            base.pop_back();
        }
        if (base.length() > 6 && base.substr(base.length() - 6) == " const") {
            suffix = " const" + suffix;
            base = base.substr(0, base.length() - 6);
        }

        std::string baseName = base;
        while((pos = baseName.find(" ")) != std::string::npos) {
           baseName.replace(pos, 1, "_");
        }
        std::string newType = "__acpp_vec_" + baseName + "_" + num;
        T = newType + suffix + T.substr(endPos + 3);
      }
    }
    return T;
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    std::error_code EC1, EC2;
    llvm::raw_fd_ostream HppFile(HppPath, EC1, llvm::sys::fs::OF_None);
    llvm::raw_fd_ostream CppFile(CppPath, EC2, llvm::sys::fs::OF_None);

    if (EC1 || EC2) {
      llvm::errs() << "Error opening output files.\n";
      return;
    }

    HppFile << "// Auto-generated Builtins Declarations\n"
            << "#pragma once\n\n"
            << "#pragma clang diagnostic push\n"
            << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n\n"
            << "typedef float __acpp_vec_float_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef float __acpp_vec_float_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef float __acpp_vec_float_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef float __acpp_vec_float_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef float __acpp_vec_float_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef float __acpp_vec_float_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef double __acpp_vec_double_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef double __acpp_vec_double_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef double __acpp_vec_double_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef double __acpp_vec_double_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef double __acpp_vec_double_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef int __acpp_vec_int_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef int __acpp_vec_int_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef int __acpp_vec_int_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef int __acpp_vec_int_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef int __acpp_vec_int_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef int __acpp_vec_int_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned int __acpp_vec_unsigned_int_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef short __acpp_vec_short_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef short __acpp_vec_short_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef short __acpp_vec_short_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef short __acpp_vec_short_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef short __acpp_vec_short_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned short __acpp_vec_unsigned_short_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef __fp16 __acpp_vec___fp16_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef __fp16 __acpp_vec___fp16_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef __fp16 __acpp_vec___fp16_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef __fp16 __acpp_vec___fp16_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef __fp16 __acpp_vec___fp16_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef __fp16 __acpp_vec___fp16_32 __attribute__((ext_vector_type(32)));\n"
            << "typedef long __acpp_vec_long_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef long __acpp_vec_long_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef long __acpp_vec_long_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef long __acpp_vec_long_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef long __acpp_vec_long_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef long long __acpp_vec_long_long_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef long long __acpp_vec_long_long_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef long long __acpp_vec_long_long_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef long long __acpp_vec_long_long_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef long long __acpp_vec_long_long_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef char __acpp_vec_char_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef char __acpp_vec_char_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef char __acpp_vec_char_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef char __acpp_vec_char_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef char __acpp_vec_char_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef signed char __acpp_vec_signed_char_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef signed char __acpp_vec_signed_char_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef signed char __acpp_vec_signed_char_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef signed char __acpp_vec_signed_char_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef signed char __acpp_vec_signed_char_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned char __acpp_vec_unsigned_char_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned char __acpp_vec_unsigned_char_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned char __acpp_vec_unsigned_char_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned char __acpp_vec_unsigned_char_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned char __acpp_vec_unsigned_char_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned long __acpp_vec_unsigned_long_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned long __acpp_vec_unsigned_long_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned long __acpp_vec_unsigned_long_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned long __acpp_vec_unsigned_long_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned long __acpp_vec_unsigned_long_16 __attribute__((ext_vector_type(16)));\n"
            << "typedef unsigned long long __acpp_vec_unsigned_long_long_2 __attribute__((ext_vector_type(2)));\n"
            << "typedef unsigned long long __acpp_vec_unsigned_long_long_3 __attribute__((ext_vector_type(3)));\n"
            << "typedef unsigned long long __acpp_vec_unsigned_long_long_4 __attribute__((ext_vector_type(4)));\n"
            << "typedef unsigned long long __acpp_vec_unsigned_long_long_8 __attribute__((ext_vector_type(8)));\n"
            << "typedef unsigned long long __acpp_vec_unsigned_long_long_16 __attribute__((ext_vector_type(16)));\n"
            << "extern \"C\" {\n";

    CppFile << "// Auto-generated Builtins Implementations\n\n"
            << "#pragma clang diagnostic push\n"
            << "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n\n";
    if (Arch == "amdgpu") {
      CppFile << "#include \"hipSYCL/sycl/libkernel/sscp/builtins/amdgpu_auto_builtins.hpp\"\n\n";
    } else if (Arch == "ptx") {
      CppFile << "#include \"hipSYCL/sycl/libkernel/sscp/builtins/ptx_auto_builtins.hpp\"\n\n";
      CppFile << "#pragma clang attribute push (__attribute__((target(\"sm_70,sm_72,sm_75,sm_80,sm_86,sm_87,sm_89,sm_90,sm_90a,sm_100,sm_100a,sm_101,sm_101a,sm_120,sm_120a,ptx62,ptx63,ptx64,ptx65,ptx70,ptx71,ptx72,ptx73,ptx74,ptx75,ptx76,ptx77,ptx78,ptx80,ptx81,ptx82,ptx83,ptx84,ptx85,ptx86,ptx87\"))), apply_to=function)\n";
    }
    CppFile << "extern \"C\" {\n";

    // Initialize builtins so they populate the IdentifierTable
    Ctx.BuiltinInfo.initializeBuiltins(Ctx.Idents, Ctx.getLangOpts());
    
    std::vector<unsigned> BuiltinIDs;
    for (auto it = Ctx.Idents.begin(); it != Ctx.Idents.end(); ++it) {
      unsigned ID = it->getValue()->getBuiltinID();
      if (ID >= Builtin::FirstTSBuiltin) {
        std::string Name = it->getKey().str();
        if (Arch == "amdgpu" && (Name.find("amdgcn") != std::string::npos || Name.find("amdgpu") != std::string::npos)) {
          BuiltinIDs.push_back(ID);
        } else if (Arch == "ptx" && Name.find("nvvm") != std::string::npos && Name.find("__builtin_ptx_") == std::string::npos && Name.find("__nvvm_compiler_") == std::string::npos && Name.find("__nvvm_mem") == std::string::npos) {
          BuiltinIDs.push_back(ID);
        }
      }
    }
    // Sort to ensure deterministic generation
    std::sort(BuiltinIDs.begin(), BuiltinIDs.end(), [&](unsigned a, unsigned b) {
      return Ctx.BuiltinInfo.getName(a) < Ctx.BuiltinInfo.getName(b);
    });

    for (unsigned ID : BuiltinIDs) {
      std::string Name(Ctx.BuiltinInfo.getName(ID));
      const char* Features = Ctx.BuiltinInfo.getRequiredFeatures(ID);
      
      if (Name.find("atomic_inc") != std::string::npos ||
          Name.find("atomic_dec") != std::string::npos ||
          Name.find("fence") != std::string::npos ||
          Name.find("div_scale") != std::string::npos ||
          Name.find("interp") != std::string::npos ||
          Name.find("buffer_rsrc") != std::string::npos ||
          Name.find("r600_") != std::string::npos) {
        continue;
      }

      ASTContext::GetBuiltinTypeError Error;
      unsigned IntegerConstantArgs = 0;
      QualType FuncType = Ctx.GetBuiltinType(ID, Error, &IntegerConstantArgs);
      
      if (Error == ASTContext::GE_None && !FuncType.isNull()) {
        const auto* FPT = FuncType->getAs<FunctionProtoType>();
        if (!FPT) continue;
        
        // Build the parameters string
        std::string ParamsStr;
        llvm::raw_string_ostream ParamsOS(ParamsStr);
        for (unsigned i = 0; i < FPT->getNumParams(); ++i) {
          if (i > 0) ParamsOS << ", ";
          ParamsOS << sanitizeType(FPT->getParamType(i).getAsString()) << " arg" << i;
        }

        std::string Sig = llvm::formatv("{0} __acpp_{1}({2})", 
            sanitizeType(FPT->getReturnType().getAsString()), Name, ParamsOS.str()).str();

        if (Sig.find("__fp16") != std::string::npos || Sig.find("__amdgpu_buffer_rsrc_t") != std::string::npos) {
          continue;
        }

        // Write declaration to .hpp
        HppFile << Sig << ";\n";
        
        // Build the arguments string
        std::string ArgsStr;
        llvm::raw_string_ostream ArgsOS(ArgsStr);
        for (unsigned i = 0; i < FPT->getNumParams(); ++i) {
          if (i > 0) ArgsOS << ", ";
          if (IntegerConstantArgs & (1 << i)) {
            ArgsOS << "1";
          } else {
            ArgsOS << "arg" << i;
          }
        }

        // Write definition to .cpp
        CppFile << "__attribute__((always_inline))\n";
        CppFile << llvm::formatv("{0} {\n  {1}{2}({3});\n}\n\n",
            Sig,
            FPT->getReturnType()->isVoidType() ? "" : "return ",
            Name,
            ArgsOS.str());
      }
    }
    
    HppFile << "}\n#pragma clang diagnostic pop\n";
    CppFile << "}\n";
    if (Arch == "ptx") {
      CppFile << "#pragma clang attribute pop\n";
    }
    CppFile << "#pragma clang diagnostic pop\n";
  }
};

class BuiltinGenAction : public ASTFrontendAction {
  std::string Arch;
  std::string HppPath;
  std::string CppPath;
public:
  BuiltinGenAction(std::string Arch, std::string Hpp, std::string Cpp) : Arch(Arch), HppPath(Hpp), CppPath(Cpp) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    return std::make_unique<BuiltinGenConsumer>(Arch, HppPath, CppPath);
  }
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <arch(amdgpu|ptx)> <output_hpp_path> <output_cpp_path>\n";
    return 1;
  }
  
  std::string Arch = argv[1];
  std::string HppPath = argv[2];
  std::string CppPath = argv[3];

  std::string TargetTriple = "amdgcn-amd-amdhsa";
  if (Arch == "ptx") {
    TargetTriple = "nvptx64-nvidia-cuda";
  }

  std::vector<std::string> args = {"-target", TargetTriple, "-nogpulib", "-fsyntax-only"};
  bool success = tooling::runToolOnCodeWithArgs(std::make_unique<BuiltinGenAction>(Arch, HppPath, CppPath), "void dummy(){}", args);
  if (!success) {
    std::cerr << "runToolOnCodeWithArgs failed!\n";
    return 1;
  }
  std::cerr << "Finished successfully.\n";
  return 0;
}

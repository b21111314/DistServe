/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "b2b_gemm_run.h"
#include "test_run.h"

////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord gemm_f16_sm75_problem_size_0(128*640, 64, 576);
cutlass::gemm::GemmCoord gemm_f16_sm75_problem_size_1(128*640, 256, 64);

bool run_nonfused_gemm_f16() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(1); //beta = 1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta = 1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2
  >;

  B2bNonFusedGemmRun<Gemm0, Gemm1> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back FP16 TN GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_f16_sm75_problem_size_0, gemm_f16_sm75_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

bool run_fused_gemm_f16_shmem() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 256, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;


  const bool SmemAccumulator = true;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    SmemAccumulator
  >;

  B2bFusedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN GEMMs with shared memory staging...\n";
  bool passed = fusedGemm.run(gemm_f16_sm75_problem_size_0, gemm_f16_sm75_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_f16,
    &run_fused_gemm_f16_shmem
  };

  return testRun(75, funcs, "gemm f16 shmem staging");


}


////////////////////////////////////////////////////////////////////////////////

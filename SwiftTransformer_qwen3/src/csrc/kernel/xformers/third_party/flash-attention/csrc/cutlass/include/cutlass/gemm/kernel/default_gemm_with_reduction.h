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

/*! \file
  \brief 
    Defines a GEMM with Reduction based on an existing UniversalGemm kernel.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/kernel/gemm_with_fused_epilogue.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/epilogue/threadblock/default_epilogue_with_reduction.h"
#include "cutlass/epilogue/threadblock/epilogue_with_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Complex elementwise transformation on A operand
  ComplexTransform TransformA,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Complex elementwise transformation on B operand
  ComplexTransform TransformB,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Operator class tag
  typename OperatorClass,
  /// Tag indicating architecture to tune for
  typename ArchTag,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Epilogue reduction operator
  typename EpilogueReductionOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Operation performed by GEMM
  typename Operator,
  ///
  typename Enable = void
>
struct DefaultGemmWithReduction {

  using GemmBase = typename DefaultGemmUniversal<
    ElementA_, LayoutA_, TransformA, kAlignmentA,
    ElementB_, LayoutB_, TransformB, kAlignmentB,
    ElementC_, LayoutC_, ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    Operator,
    SharedMemoryClearOption::kClearLastStage
  >::GemmKernel;

  // Define epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementC_,
    EpilogueOutputOp,
    EpilogueReductionOp,
    GemmBase::Epilogue::kElementsPerAccess
  >::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = GemmWithFusedEpilogue<
    typename GemmBase::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization: ArchTag = cutlass::arch::Sm70
///
///
template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Complex elementwise transformation on A operand
  ComplexTransform TransformA,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Complex elementwise transformation on B operand
  ComplexTransform TransformB,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Operator class tag
  typename OperatorClass,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Epilogue reduction operator
  typename EpilogueReductionOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Operation performed by GEMM
  typename Operator,
  ///
  typename Enable
>
struct DefaultGemmWithReduction<
  ElementA_, LayoutA_, TransformA, kAlignmentA, 
  ElementB_, LayoutB_, TransformB, kAlignmentB,
  ElementC_, LayoutC_,
  ElementAccumulator,
  OperatorClass,
  cutlass::arch::Sm70,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  EpilogueReductionOp,
  ThreadblockSwizzle,
  Stages,
  Operator,
  Enable
  >  {

  using GemmBase = typename DefaultGemmUniversal<
    ElementA_, LayoutA_, TransformA, kAlignmentA,
    ElementB_, LayoutB_, TransformB, kAlignmentB,
    ElementC_, LayoutC_, ElementAccumulator,
    OperatorClass,
    cutlass::arch::Sm70,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    Operator
  >::GemmKernel;

  // Define epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithReductionVoltaTensorOp<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementC_,
    EpilogueOutputOp,
    EpilogueReductionOp,
    GemmBase::Epilogue::kElementsPerAccess
  >::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = GemmWithFusedEpilogue<
    typename GemmBase::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

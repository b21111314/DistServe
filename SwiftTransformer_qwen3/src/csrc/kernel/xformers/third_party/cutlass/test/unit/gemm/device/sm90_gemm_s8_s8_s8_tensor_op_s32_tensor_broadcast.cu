/***************************************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface with an elementwise tensor-tensor broadcast epilogue
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/epilogue_tensor_broadcast.hpp"
#include "cutlass/epilogue/thread/linear_combination_tensor_broadcast.hpp"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_tensor_broadcast.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_s8t_s8n_s8n_tensor_op_gmma_s32_tensor_broadcast, 128x128x128_2x2x1_ActReLU_Bin0Mul_Bin1Plus_UnaryHardSwish) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementOutput = int32_t;
  using ElementAccumulator = ElementOutput;
  using ElementCompute = ElementOutput;
  using ElementBias = ElementOutput;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      int8_t, LayoutA, 16,
      int8_t, LayoutB, 16,
      int32_t,
      Shape<_128,_128,_128>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<
        ElementOutput, ElementAccumulator, ElementCompute, ElementBias,
        cutlass::epilogue::thread::ReLu,
        cutlass::multiplies,
        cutlass::plus,
        cutlass::epilogue::thread::HardSwish
        >,
      cutlass::gemm::EpilogueDefault>>;

  EXPECT_TRUE(EpilogueOp::IsBinaryOp0Enabled);
  EXPECT_TRUE(EpilogueOp::IsBinaryOp1Enabled);
  EXPECT_TRUE(EpilogueOp::IsUnaryOpEnabled);

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllTensorBroadcast<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

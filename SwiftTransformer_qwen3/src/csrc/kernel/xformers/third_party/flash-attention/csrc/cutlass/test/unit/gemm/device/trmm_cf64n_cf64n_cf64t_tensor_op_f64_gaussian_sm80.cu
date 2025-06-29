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
    \brief Tests for device-wide TRMM interface

  
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/blas3.h"
#include "cutlass/gemm/device/trmm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/trmm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_trmm_universal.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_cf64n_cf64n_cf64t_ls_u_nu_tensor_op_f64_gaussian, 32x32x16_16x16x16) {

  using ElementOutput = cutlass::complex<double>;
  using ElementAccumulator = cutlass::complex<double>;

  using Trmm = cutlass::gemm::device::Trmm<
    cutlass::complex<double>,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    cutlass::complex<double>,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddGaussianComplex,
    cutlass::ComplexTransform::kNone
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_cf64h_cf64n_cf64t_ls_u_nu_tensor_op_f64_gaussian, 64x64x16_32x32x16) {

  using ElementOutput = cutlass::complex<double>;
  using ElementAccumulator = cutlass::complex<double>;

  using Trmm = cutlass::gemm::device::Trmm<
    cutlass::complex<double>,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    cutlass::complex<double>,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddGaussianComplex,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

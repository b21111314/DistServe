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
    \brief Tests for device-wide GEMM interface
    
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_complex.h"

    
#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 32x64x8_32x64x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    4
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 64x64x8_32x64x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 128x128x8_32x64x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 64x128x8_32x64x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 128x64x8_64x32x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 128x128x8_64x64x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

TEST(SM80_Device_Gemm_cf32t_cf32n_cf32t_simt_cf32, 128x256x8_64x64x1) {
  
  using Element = cutlass::complex<float>;

  using Gemm = cutlass::gemm::device::GemmComplex<
    Element, 
    cutlass::layout::RowMajor,
    Element, 
    cutlass::layout::ColumnMajor,
    Element,
    cutlass::layout::RowMajor, 
    Element,
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 
        1,
        Element, 
        Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kConjugate
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemmComplex<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

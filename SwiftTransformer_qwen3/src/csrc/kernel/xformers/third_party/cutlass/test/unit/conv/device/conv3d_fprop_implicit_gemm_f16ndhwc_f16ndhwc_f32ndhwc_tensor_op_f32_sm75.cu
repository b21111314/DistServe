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
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv3d_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Conv3d_Fprop_Analytic_ImplicitGemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32,
  128x128_32x3_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv3dFpropKernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
    ElementA, cutlass::layout::TensorNDHWC,
    ElementB, cutlass::layout::TensorNDHWC,
    ElementC, cutlass::layout::TensorNDHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd
  >::Kernel;

  using Conv3dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv3dFpropKernel>;

  /// Run all unit test sizes with device-level Conv3d instance
  EXPECT_TRUE(test::conv::device::TestAllConv3d<Conv3dFprop>());
}

////////////////////////////////////////////////////////////////////////////////
#endif  // CUTLASS_ARCH_MMA_SM75_SUPPORTED

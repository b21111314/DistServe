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
    \brief Unit tests for thread-level GEMM
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"

#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"

#include "cutlass/epilogue/thread/linear_combination.h"    
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued single precision tests
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_f32_32x64_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;
  
  using Shape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using Element = float;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<4, 4, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_f32_32x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using Element = float;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<4, 4, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_f32_64x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using Element = float;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<4, 4, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_f32_128x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using Element = float;
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<4, 4, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued double precision tests
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_f64_32x64_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = double;
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

TEST(SM50_Epilogue_threadblock_epilogue, simt_f64_32x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = double;
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;

  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

TEST(SM50_Epilogue_threadblock_epilogue, simt_f64_64x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = double;
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;

  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

TEST(SM50_Epilogue_threadblock_epilogue, simt_f64_128x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = double;
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;

  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Complex-valued single-precision
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_complex_f32_32x64_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::complex<float>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_complex_f32_32x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::complex<float>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

TEST(SM50_Epilogue_threadblock_epilogue, simt_complex_f32_128x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::complex<float>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Complex-valued double-precision
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_complex_f64_32x64_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::complex<double>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<1, 1, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_complex_f64_32x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::complex<double>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<1, 1, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

TEST(SM50_Epilogue_threadblock_epilogue, simt_complex_f64_128x128_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::complex<double>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;

  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<1, 1, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Quaternion-valued single-precision
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_Epilogue_threadblock_epilogue, simt_quaternion_f32_32x64_32x64x8) {

  //
  // Define the warp-level matrix multiply
  //

  using Element = cutlass::Quaternion<float>;
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  int const kElementsPerAccess = 1;

  using Shape = cutlass::gemm::GemmShape<32, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  
  using ElementC = ElementAccumulator;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;

  using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<4, 8>,
      cutlass::layout::RowMajorInterleaved<2>,
      cutlass::gemm::GemmShape<2, 2, 1>
    >
  >;

  //
  // Output operator
  //

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator,
    ElementCompute
  >;

  //
  // Define the epilogue
  //

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

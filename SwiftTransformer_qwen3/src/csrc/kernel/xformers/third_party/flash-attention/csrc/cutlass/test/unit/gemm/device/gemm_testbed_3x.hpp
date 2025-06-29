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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

#include "testbed_utils.h"

#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cute/int_tuple.hpp"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail{

// Helper classes that take default data type when 
// the Gemm::EpilogueOutputOp does not have ElementCompute
// and ElementScalar. 
// (e.g. when Sm90TreeVisitor is used as FusionCallbacks)
template <typename Gemm, typename Default, typename = void>
struct ElementComputeType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementComputeType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementCompute>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementCompute;
};

template <typename Gemm, typename Default, typename = void>
struct ElementScalarType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementScalarType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementScalar>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementScalar;
};



// The number of splits to test.
//
// This class makes it harder to confuse the order of arguments
// of the various run(...) functions in this file.  The constructor
// is explicit, so one can't just type 42 (or false, which the
// compiler unhelpfully turns into 0); one has to type Splits(42).
// Splits() picks the default number of splits, 1.
//
// The conversion-to-int operator (operator int()) MUST be explicit!
// Conversion to int MUST require static_cast<int>.
// Otherwise, that defeats a key purpose of this class,
// which is to catch common errors of confusing the order
// of function arguments.
class Splits {
public:
  Splits() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !std::is_same_v<IntegralNotBool, bool>)) >
  explicit Splits(IntegralNotBool splits) : splits_(splits) {}
  explicit operator int() const { return splits_; }
private:
  int splits_ = 1;
};

// The number of iterations to test.
//
// This class, like Splits above makes it harder to confuse
// the order of arguments of the various run(...) functions in this file.
// Iterations() picks the default number of iterations, 20.
class Iterations {
public:
  Iterations() = default;

  template<class IntegralNotBool,
    __CUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !std::is_same_v<IntegralNotBool, bool>)) >
  explicit Iterations(IntegralNotBool iterations) : iterations_(iterations) {}
  explicit operator int() const { return iterations_; }
private:
  int iterations_ = 20;
};

template <
  typename Gemm,
  template <class T> class ActivationFunctor_ = cutlass::epilogue::thread::Identity
>
struct TestbedImpl {
  // Kernel data types
  using ElementA = typename Gemm::GemmKernel::ElementA;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using ElementB = typename Gemm::GemmKernel::ElementB;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ElementC = std::conditional_t<std::is_void_v<typename Gemm::GemmKernel::ElementC>,
                    typename Gemm::GemmKernel::ElementD,typename Gemm::GemmKernel::ElementC>;
  using StrideC  = typename Gemm::GemmKernel::StrideC;
  using ElementD = typename Gemm::GemmKernel::ElementD;
  using StrideD  = typename Gemm::GemmKernel::StrideD;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
  /// For custom EVTs
  using ElementCompute = typename ElementComputeType<Gemm, ElementAccumulator>::Type;
  using ElementScalar = typename ElementScalarType<Gemm, ElementCompute>::Type;
  using ActivationFunctor = ActivationFunctor_<ElementCompute>;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static constexpr uint32_t mma_promotion_interval = 4;

  // Looks at Cute Stride to check Row / Column Major
  template<typename Stride>
  static constexpr bool is_row_or_col_major(){
    int stride_0 = int(cute::size<0>(Stride{}));
    int stride_1 = int(cute::size<1>(Stride{}));
    int depth = cute::depth(Stride{});
    return ((stride_0 == 1) || (stride_1 == 1)) && (depth == 1);
  }

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagA = cutlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = cutlass::detail::StrideToLayoutTagA_t<StrideC>;
  using LayoutTagD = cutlass::detail::StrideToLayoutTagA_t<StrideD>;

  /// Initialization
  StrideA stride_a;
  StrideB stride_b;
  StrideC stride_c;
  StrideD stride_d;
  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutTagD> reference_D;
  uint32_t sm_count;

  // Used to force multi-wave tests for persistent kernel schedules
  constexpr static int MaxSmCount = 16;
  //
  // Methods
  //

  TestbedImpl(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ):
    stride_factor_A(typename LayoutTagA::Stride()),
    stride_factor_B(typename LayoutTagB::Stride()),
    stride_factor_C(typename LayoutTagC::Stride()),
    stride_factor_D(typename LayoutTagD::Stride()),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  TestbedImpl(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_C(stride_factor_C_),
    stride_factor_D(stride_factor_D_),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {
      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<ElementD>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      }
      else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      }
      else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      }
      else {
        scope_max = 8;
        scope_min = -8;
      }
      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }

    else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);
    }

    else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }

    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }

    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }

    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = cutlass::make_Coord(M * L, K);
    auto c_coord = cutlass::make_Coord(M * L, N);
    // Cutlass has Row/Col major refers to MxK times KxN matrix product, 
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = cutlass::make_Coord(K, N * L);
    

    tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
    tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
    reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2020));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);
    tensor_C.host_view().at({0, 0}) = ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta)
  {
    auto [M, N, K, L] = problem_shape_MNKL;

    tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    }

    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    }

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L
        << ", alpha: " << float(alpha) << ", beta: " << float(beta) << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\n\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      ProblemShapeType problem_size,
      ElementScalar alpha,
      ElementScalar beta) 
  {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    auto A = cute::make_tensor(tensor_A.host_data(),
        cute::make_layout(cute::make_shape(M, K, L), stride_a));
    auto B = cute::make_tensor(tensor_B.host_data(),
        cute::make_layout(cute::make_shape(N, K, L), stride_b));
    auto C = cute::make_tensor(tensor_C.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), stride_c));
    auto D = cute::make_tensor(reference_D.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
    auto Bias = cute::make_tensor(static_cast<ElementCompute*>(nullptr),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Aux = cute::make_tensor(static_cast<ElementD*>(nullptr),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
    auto Valpha = cute::make_tensor(static_cast<ElementCompute*>(nullptr),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Vbeta = cute::make_tensor(static_cast<ElementCompute*>(nullptr),
        cute::make_layout(cute::make_shape(M, cute::_1{})));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    cutlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(C),
        decltype(D),
        decltype(Bias),
        decltype(Aux),
        decltype(Valpha),
        decltype(Vbeta),
        ActivationFunctor
        >
        epilogue_params{
          alpha, beta,
          C, D, Bias, Aux
          , Valpha, Vbeta
        };

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    return compare_reference(problem_shape_MNKL, alpha, beta);
  }

	/// Determine if the CUDA device is sufficient to run the kernel
  bool sufficient() {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    int smem_size = Gemm::GemmKernel::SharedStorageSize;

    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    cudaDeviceProp properties;
    result = cudaGetDeviceProperties(&properties, device_idx);
    this->sm_count = properties.multiProcessorCount;

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      return false;
    }

    return true;
  }

  bool profile(
    ProblemShapeType problem_size,
    int iterations,
    Gemm& gemm_op,
    typename Gemm::Arguments& arguments,
    cutlass::device_memory::allocation<uint8_t>& workspace) {
    int M = cute::size<0>(problem_size);
    int N = cute::size<1>(problem_size);
    int K = cute::size<2>(problem_size);
    int L = 1;
    if constexpr(cute::rank(ProblemShapeType{}) == 4) {
      L = cute::size<3>(problem_size);
    }


    cutlass::Status status;
    //
    // Run the GEMM
    //
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {
      status = gemm_op(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        return false;
      }
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    bool profiling = false,
    detail::Iterations iterations = Iterations{},
    detail::Splits splits = Splits{})
  {
    // Fail test if insufficient CUDA device
    if (!sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }

    this->initialize(problem_size);

    //
    // Initialize the GEMM operator
    //

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      this->sm_count = min(MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = this->sm_count;
    }
    else {
      this->sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = this->sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (std::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits) };
    }

    // DefaultEpilogue
    auto arguments = typename Gemm::Arguments {
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        tensor_A.device_data(), stride_a,
        tensor_B.device_data(), stride_b
      },
      {
        {alpha, beta},
        tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
      },
      hw_info,
      scheduler_args
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
      return profile(problem_size, static_cast<int>(iterations), gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }

      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

      //
      // Verify
      //
      bool passed = this->verify(problem_size, alpha, beta);
      if (!passed) {
        std::cout << "Error : Failed : with alpha: " << float(alpha) << ", beta: " << float(beta)
                  << "\n";
      }

      return passed;
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  template <class T> class ActivationFunctor
>
struct Testbed3x {

  using TestBedImpl = typename detail::TestbedImpl<Gemm, ActivationFunctor>;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using ElementAccumulator   = typename TestBedImpl::ElementAccumulator;
  using ElementCompute       = typename TestBedImpl::ElementCompute;
  using ElementScalar        = typename TestBedImpl::ElementScalar;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;
  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;

  // Detail Implementation
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x(
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = TestBedImpl::kDefaultSeed)
      : impl_(init_A_, init_B_, init_C_, seed_) {}

  Testbed3x(    
      typename LayoutTagA::Stride stride_factor_A_,
      typename LayoutTagB::Stride stride_factor_B_,
      typename LayoutTagC::Stride stride_factor_C_,
      typename LayoutTagD::Stride stride_factor_D_,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
          uint64_t seed_ = TestBedImpl::kDefaultSeed)
      : impl_(stride_factor_A_,
              stride_factor_B_,
              stride_factor_C_,
              stride_factor_D_,
              init_A_,
              init_B_,
              init_C_,
              seed_) {}

  /// Executes one test
  bool run(
    typename TestBedImpl::ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    detail::Splits splits = detail::Splits{},
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{})
  {
    return impl_.run(
        problem_size, alpha, beta, profiling, iterations, splits
        );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Testbed for GEMMs with fused epilogues using the fusion::FusionOperation API
// Does not support testing of custom EVTs
template <typename Gemm>
struct Testbed3xFusionOperation {

  using TestBedImpl = typename detail::TestbedImpl<Gemm>;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;
  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;
  using LayoutTagScalar = cutlass::layout::PackedVectorLayout; // scalars are size-1 vectors
  using LayoutTagVector = cutlass::layout::PackedVectorLayout;

  using ElementA = typename Kernel::ElementA;
  using StrideA  = typename Kernel::StrideA;
  using ElementB = typename Kernel::ElementB;
  using StrideB  = typename Kernel::StrideB;
  using ElementC = typename Kernel::ElementC;
  using StrideC  = typename Kernel::StrideC;
  using ElementD = typename Kernel::ElementD;
  using StrideD  = typename Kernel::StrideD;
  using ProblemShapeType   = typename Kernel::ProblemShape;
  using ElementAccumulator = typename Kernel::ElementAccumulator;
  
  //
  // FusionOperation derived types/queries
  //
  using FusionOp = typename Gemm::EpilogueOutputOp;
  static_assert(cute::is_base_of_v<cutlass::epilogue::fusion::FusionOperation, FusionOp>);

  // fusion types are potentially void if the fusion is not supported
  // helper so we don't try to construct HostTensor with void type
  template <typename T, typename U = uint8_t>
  using non_void_t = cute::conditional_t<cute::is_void_v<T>, U, T>;

  using ElementCompute    = typename FusionOp::ElementCompute;
  using ElementScalar     = typename FusionOp::ElementScalar;
  using ElementBias       = non_void_t<typename FusionOp::ElementBias>;
  using ElementAux        = non_void_t<typename FusionOp::ElementAux>;
  using ElementAmax       = non_void_t<typename FusionOp::ElementAmax>;
  using LayoutTagAux      = non_void_t<typename FusionOp::GmemLayoutTagAux, LayoutTagD>;
  using ActivationFunctor = non_void_t<typename FusionOp::ActivationFn,
                              cutlass::epilogue::thread::Identity<ElementCompute>>;

  static constexpr bool IsBiasEnabled        = FusionOp::IsPerRowBiasSupported;
  static constexpr bool IsPerRowScaleEnabled = FusionOp::IsPerRowScaleSupported;
  static constexpr bool IsScaleFactorEnabled = FusionOp::IsScaleFactorSupported;
  static constexpr bool IsAuxEnabled         = FusionOp::IsAuxOutSupported;
  static constexpr bool IsAbsMaxEnabled      = FusionOp::IsAbsMaxSupported;

  // Legacy support for deprecated bias-elementwise collective, will be removed next release
  using EpiloguePolicy = typename Epilogue::DispatchPolicy;
  static constexpr bool IsLegacy =
    cute::is_same_v<
      EpiloguePolicy,
      cutlass::epilogue::Sm90TmaWarpSpecializedBiasElementwise<
        EpiloguePolicy::StagesC, EpiloguePolicy::StagesD, EpiloguePolicy::FragmentSize>
    >;

  // Inputs
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> alpha;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> beta;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_A;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_B;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_C;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_D;
  cutlass::HostTensor<ElementScalar, LayoutTagScalar> scale_Aux;
  cutlass::HostTensor<ElementBias  , LayoutTagVector> bias;
  // Outputs
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> abs_max_D;
  cutlass::HostTensor<ElementAux , LayoutTagAux   > tensor_Aux;
  cutlass::gemm::TagToStrideC_t<   LayoutTagAux   > stride_Aux;
  // References
  cutlass::HostTensor<ElementAux , LayoutTagAux   > reference_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAmax, LayoutTagScalar> reference_abs_max_D;

  // Detail Implementation
  TestBedImpl impl_;

  // Whether to use relative equality checks
  bool check_relative_equality = false;
  // Are scalars copied to device memory before kernel launch
  bool use_device_scalars = false;
  // If per-row scale is enabled and this is true, beta is passed as a host scalar instead of device vector
  bool disable_vector_beta = false;
  // Random distribution with which to initialize the A/B/C/D/Aux scaling factors
  cutlass::Distribution::Kind init_scale = cutlass::Distribution::Uniform;
  // Random distribution with which to initialize the bias vector
  cutlass::Distribution::Kind init_bias = cutlass::Distribution::Uniform;

  // Factors used for calculating relative equality. These default
  // values are borrowed from those used by default in the CUTLASS
  // profiler for performing relative equality checks.
  float epsilon = 0.05f;
  float nonzero_floor = 1.0f / 256.0f;

  //
  // Methods
  //
  Testbed3xFusionOperation(
    bool check_relative_equality_ = false,
    bool use_device_scalars_ = false,
    bool disable_vector_beta_ = false,
    cutlass::Distribution::Kind init_scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_bias_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) : impl_(init_A_, init_B_, init_C_, seed_),
      check_relative_equality(check_relative_equality_),
      use_device_scalars(use_device_scalars_),
      init_scale(init_scale_), init_bias(init_bias_) { }

  Testbed3xFusionOperation(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImpl::kDefaultSeed
  ) : impl_(stride_factor_A_,
            stride_factor_B_,
            stride_factor_C_,
            stride_factor_D_,
            init_A_,
            init_B_,
            init_C_,
            seed_) { }

  /// Initializes data structures
  void initialize(ProblemShapeType problem_size, ElementScalar alpha_=1.f, ElementScalar beta_=0.f) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    auto scalar_coord = cutlass::make_Coord(1);
    auto col_vector_coord = cutlass::make_Coord(M);

    // Allocate the GEMM workspace for A/B/C/D tensor
    impl_.initialize(problem_size);

    if constexpr (IsPerRowScaleEnabled) {
      alpha.resize(col_vector_coord);
      EXPECT_TRUE(impl_.initialize_tensor(alpha.host_view(), init_scale, impl_.seed + 2023));
      if (disable_vector_beta) {
        beta.resize(scalar_coord, false);
        cutlass::reference::host::TensorFill(beta.host_view(), beta_);
      }
      else {
        beta.resize(col_vector_coord);
        EXPECT_TRUE(impl_.initialize_tensor(beta.host_view(), init_scale, impl_.seed + 2024));
      } 
    }
    else {
      alpha.resize(scalar_coord, use_device_scalars);
      beta.resize(scalar_coord, use_device_scalars);
      cutlass::reference::host::TensorFill(alpha.host_view(), alpha_);
      cutlass::reference::host::TensorFill(beta.host_view(), beta_);
    }
    alpha.sync_device();
    beta.sync_device();

    if constexpr (IsScaleFactorEnabled) {
      scale_A.resize(scalar_coord, use_device_scalars);
      scale_B.resize(scalar_coord, use_device_scalars);
      scale_C.resize(scalar_coord, use_device_scalars);
      scale_D.resize(scalar_coord, use_device_scalars);
      EXPECT_TRUE(impl_.initialize_tensor(scale_A.host_view(), init_scale, impl_.seed + 2023));
      EXPECT_TRUE(impl_.initialize_tensor(scale_B.host_view(), init_scale, impl_.seed + 2024));
      EXPECT_TRUE(impl_.initialize_tensor(scale_C.host_view(), init_scale, impl_.seed + 2025));
      EXPECT_TRUE(impl_.initialize_tensor(scale_D.host_view(), init_scale, impl_.seed + 2026));
      scale_A.sync_device();
      scale_B.sync_device();
      scale_C.sync_device();
      scale_D.sync_device();
    }

    if constexpr (IsBiasEnabled) {
      bias.resize(col_vector_coord);
      EXPECT_TRUE(impl_.initialize_tensor(bias.host_view(), init_bias, impl_.seed + 2023));
      bias.sync_device();
    }

    if constexpr (IsAbsMaxEnabled) {
      abs_max_D.resize(scalar_coord);
      abs_max_D.sync_device();
      reference_abs_max_D.resize(scalar_coord);
    }

    if constexpr (IsAuxEnabled) {
      auto aux_coord = cutlass::make_Coord(M * L, N);
      auto aux_layout = cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(aux_coord, typename LayoutTagAux::Stride{});
      tensor_Aux.resize(aux_coord, aux_layout);
      reference_Aux.resize(aux_coord, aux_layout, false);
      tensor_Aux.sync_device();
      stride_Aux = cutlass::make_cute_packed_stride(cutlass::gemm::TagToStrideC_t<LayoutTagAux>{}, cute::make_shape(M, N, L));

      if constexpr (IsScaleFactorEnabled) {
        scale_Aux.resize(scalar_coord, use_device_scalars);
        EXPECT_TRUE(impl_.initialize_tensor(scale_Aux.host_view(), init_scale, impl_.seed + 2027));
        scale_Aux.sync_device();
      }

      if constexpr (IsAbsMaxEnabled) {
        abs_max_Aux.resize(scalar_coord);
        abs_max_Aux.sync_device();
        reference_abs_max_Aux.resize(scalar_coord);
      }
    }

  }

  template <
    class Element,
    class Layout
  >
  bool equality_check(
    cutlass::TensorView<Element, Layout> const& lhs,
    cutlass::TensorView<Element, Layout> const& rhs) const {

    if (check_relative_equality) {
      return cutlass::reference::host::TensorRelativelyEquals(
        lhs, rhs, Element(epsilon), Element(nonzero_floor));
    }
    else {
      return cutlass::reference::host::TensorEquals(lhs, rhs);
    }
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(cute::Shape<int,int,int,int> problem_shape_MNKL) {
    auto [M, N, K, L] = problem_shape_MNKL;
    auto coord_0 = cutlass::make_Coord(0);

    EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.tensor_C.host_view()), 0);

    impl_.tensor_D.sync_host();
    if (impl_.tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.tensor_D.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(impl_.reference_D.host_view()), 0);
    }
    bool passed = equality_check(impl_.reference_D.host_view(), impl_.tensor_D.host_view());

    if constexpr (IsAbsMaxEnabled) {
      abs_max_D.sync_host();
      passed &= equality_check(reference_abs_max_D.host_view(), abs_max_D.host_view());
    }

    if constexpr (IsAuxEnabled) {
      tensor_Aux.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_Aux.host_view()), 0);
      passed &= equality_check(reference_Aux.host_view(), tensor_Aux.host_view());
      if constexpr (IsAbsMaxEnabled) {
        abs_max_Aux.sync_host();
        passed &= equality_check(reference_abs_max_Aux.host_view(), abs_max_Aux.host_view());
      }
    }

    EXPECT_TRUE(passed);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L;
      if constexpr (IsScaleFactorEnabled) {
        file
          << ", scale_a: " << scale_A.at(coord_0)
          << ", scale_b: " << scale_B.at(coord_0)
          << ", scale_c: " << scale_C.at(coord_0);
      }
      if constexpr (IsPerRowScaleEnabled) {
        file << "\n\nvalpha = \n" << alpha.host_view();
        file << "\n\nvbeta = \n" << beta.host_view();
      } else {
        file
          << ", alpha: " << alpha.at(coord_0) << ", beta: " << beta.at(coord_0);
      }
      file << "\n\n";

      if constexpr (IsAbsMaxEnabled) {
        file << "scale_d: " << float(scale_D.at(coord_0));
        file << "\nReference abs_max_D :";
        file << " " << float(reference_abs_max_D.at(coord_0));

        file << "\nComputed abs_max_D :";
        file << " " << float(abs_max_D.at(coord_0));
        file << "\n\n";
        if constexpr (IsAuxEnabled) {
          file << "scale_aux: " << float(scale_Aux.at(coord_0));
          file << "\nReference abs_max_Aux :";
          file << " " << float(reference_abs_max_Aux.at(coord_0));

          file << "\nComputed abs_max_Aux :";
          file << " " << float(abs_max_Aux.at(coord_0));     
          file << "\n\n";
        }
      }

      file
        << "A =\n" << impl_.tensor_A.host_view()
        << "\nB =\n" << impl_.tensor_B.host_view()
        << "\nC =\n" << impl_.tensor_C.host_view();

      if constexpr (IsBiasEnabled) {
        file << "\n\nBias = \n" << bias.host_view();
      }

      if constexpr (IsAuxEnabled) {
        file
          << "\n\nReference Aux =\n" << reference_Aux.host_view()
          << "\n\nComputed Aux =\n" << tensor_Aux.host_view();
      }
      file
        << "\n\nReference D =\n" << impl_.reference_D.host_view()
        << "\n\nComputed D =\n" << impl_.tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result against a reference implementation
  bool verify(ProblemShapeType problem_size)
  {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::get<0>(problem_shape_MNKL);
    auto N = cute::get<1>(problem_shape_MNKL);
    auto K = cute::get<2>(problem_shape_MNKL);
    auto L = cute::get<3>(problem_shape_MNKL);
    auto coord_0 = cutlass::make_Coord(0);

    auto A = cute::make_tensor(impl_.tensor_A.host_data(),
        cute::make_layout(cute::make_shape(M, K, L), impl_.stride_a));
    auto B = cute::make_tensor(impl_.tensor_B.host_data(),
        cute::make_layout(cute::make_shape(N, K, L), impl_.stride_b));
    auto C = cute::make_tensor(impl_.tensor_C.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), impl_.stride_c));
    auto D = cute::make_tensor(impl_.reference_D.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), impl_.stride_d));
    auto Bias = cute::make_tensor(bias.host_data(),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Aux = cute::make_tensor(reference_Aux.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), stride_Aux));
    auto Valpha = cute::make_tensor(alpha.host_data(),
        cute::make_layout(cute::make_shape(M, cute::_1{})));
    auto Vbeta = cute::make_tensor(beta.host_data(),
        cute::make_layout(cute::make_shape(M, cute::_1{})));

    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      decltype(Bias),
      decltype(Aux),
      decltype(Valpha),
      decltype(Vbeta),
      ActivationFunctor>
        epilogue_params{};

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha.at(coord_0);
    epilogue_params.beta = beta.at(coord_0);

    if constexpr (IsScaleFactorEnabled) {
      epilogue_params.scale_a = scale_A.at(coord_0);
      epilogue_params.scale_b = scale_B.at(coord_0);
      epilogue_params.scale_c = scale_C.at(coord_0);
      epilogue_params.scale_d = scale_D.at(coord_0);
    }

    if constexpr (IsBiasEnabled) {
      epilogue_params.Bias = Bias;
    }

    if constexpr (IsAbsMaxEnabled) {
      epilogue_params.abs_max_D = reference_abs_max_D.host_data();
    }

    if constexpr (IsAuxEnabled) {
      epilogue_params.Aux = Aux;
      if constexpr (IsScaleFactorEnabled) {
        epilogue_params.scale_aux = scale_Aux.at(coord_0);
      }
      if constexpr (IsAbsMaxEnabled) {
        epilogue_params.abs_max_Aux = reference_abs_max_Aux.host_data();
      }
    }

    if constexpr (IsPerRowScaleEnabled) {
      epilogue_params.Valpha = Valpha;
      if (not disable_vector_beta) {
        epilogue_params.Vbeta = Vbeta;
      }
    }

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    return compare_reference(problem_shape_MNKL);
  }

  /// Executes one test
  bool run(
    ProblemShapeType problem_size,
    ElementScalar alpha_ = ElementScalar(1),
    ElementScalar beta_ = ElementScalar(0),
    detail::Splits splits = detail::Splits{},
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{})
  {
    // Fail test if insufficient CUDA device
    if (!impl_.sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }
    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      impl_.sm_count = min(impl_.MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = impl_.sm_count;
    }
    else {
      impl_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = impl_.sm_count;
    }

    /// Initializes data structures
    /// A/B/C/D Tensor
    initialize(problem_size, alpha_, beta_);

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (std::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits) };
    }

    arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        impl_.tensor_A.device_data(), impl_.stride_a,
        impl_.tensor_B.device_data(), impl_.stride_b
      },
      { // Epilogue arguments
        {}, // thread
        impl_.tensor_C.device_data(),
        impl_.stride_c,
        impl_.tensor_D.device_data(),
        impl_.stride_d
      }, // Epilogue arguments end
      hw_info,
      scheduler_args
    };

    auto coord_0 = cutlass::make_Coord(0);
    if constexpr (IsLegacy) {
      arguments.epilogue.thread = {
        alpha.at(coord_0),
        beta.at(coord_0),
        alpha.device_data(),
        beta.device_data()
      };
      arguments.epilogue.ptr_Bias = bias.device_data();
      arguments.epilogue.ptr_T = tensor_Aux.device_data();
    }
    else {
      auto &fusion_args = arguments.epilogue.thread;
      
      fusion_args.alpha = alpha.at(coord_0);
      fusion_args.beta = beta.at(coord_0);
      fusion_args.alpha_ptr = alpha.device_data();
      fusion_args.beta_ptr = beta.device_data(); // if disable_vector_beta is true this is nullptr

      if constexpr (IsScaleFactorEnabled) {
        fusion_args.scale_a = scale_A.at(coord_0);
        fusion_args.scale_b = scale_B.at(coord_0);
        fusion_args.scale_c = scale_C.at(coord_0);
        fusion_args.scale_d = scale_D.at(coord_0);
        fusion_args.scale_a_ptr = scale_A.device_data();
        fusion_args.scale_b_ptr = scale_B.device_data();
        fusion_args.scale_c_ptr = scale_C.device_data();
        fusion_args.scale_d_ptr = scale_D.device_data();
      }

      if constexpr (IsBiasEnabled) {
        fusion_args.bias_ptr = bias.device_data();
      }

      // example of how to set kernel activation arguments
      if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledGELU_taylor<ElementCompute>>) {
        // see ActivationFunctor::Arguments in activation.h for definition
        // if Arguments doesn't exist then fusion_args.activation is empty
        fusion_args.activation.scale = ElementCompute(1);
      }

      if constexpr (IsAbsMaxEnabled) {
        fusion_args.amax_D_ptr = abs_max_D.device_data();
      }

      if constexpr (IsAuxEnabled) {
        fusion_args.aux_ptr = tensor_Aux.device_data();
        fusion_args.dAux = stride_Aux;
        if constexpr (IsScaleFactorEnabled) {
          fusion_args.scale_aux = scale_Aux.at(coord_0);
          fusion_args.scale_aux_ptr = scale_Aux.device_data();
        }
        if constexpr (IsAbsMaxEnabled) {
          fusion_args.amax_aux_ptr = abs_max_Aux.device_data();
        }
      }
    }

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
      return impl_.profile(problem_size, static_cast<int>(iterations), gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }

      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

      //
      // Verify
      //
      bool passed = this->verify(problem_size);
      if (!passed) {
        std::cout << "Error : Failed : with alpha: " << float(alpha_) << ", beta: " << float(beta_)
                  << "\n";
      }

      return passed;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  typename Testbed = Testbed3x<Gemm, cutlass::epilogue::thread::Identity>
>
bool TestAll(double alpha = 1.0, double beta = 0.0, Testbed testbed = {}) {
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  if constexpr (std::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};

  std::vector<int> problem_splits = {1};
  if constexpr (std::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
    problem_splits.push_back(2);
    problem_splits.push_back(3);

    // As many splits as there are maximum k tiles
    problem_splits.push_back(Stages + 1);
  }

  bool passed = true;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int splits : problem_splits) {
          ProblemShapeType problem_size;
          if constexpr (cute::rank(ProblemShapeType{}) == 4) {
            problem_size = ProblemShapeType{m, n, k, /* l */ 1};
          }
          else {
            problem_size = ProblemShapeType{m, n, k};
          }

          passed = testbed.run(
            problem_size,
            cutlass::from_real<ElementScalar>(alpha),
            cutlass::from_real<ElementScalar>(beta),
            detail::Splits(splits)
          );

          if (!passed) {
            return false;
          }
        }
      }
    }
  }

  // if we do support batched GEMM, just run one test on it to save on test time
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{256 + max_alignment, 256 + max_alignment, 160 + max_alignment, /* l */ 3};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(alpha),
      cutlass::from_real<ElementScalar>(beta)
    );

    if (!passed) {
      return false;
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestAllBiasElementwise(double alpha = 1.0, double beta = 0.0, bool check_relative_equality=false) {
  Testbed3xFusionOperation<Gemm> testbed(check_relative_equality);

  return TestAll<Gemm>(alpha, beta, testbed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestGemmPerf3x(int iterations = 20) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalar = ElementAccumulator;
  bool passed = true;

  std::vector<int> problem_size_m = { 4608 };
  std::vector<int> problem_size_n = { 4608 };
  std::vector<int> problem_size_k = { 8192 };

  Testbed3x<Gemm, cutlass::epilogue::thread::Identity> testbed;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (cute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }

        passed = testbed.run(
          problem_size,
          cutlass::from_real<ElementScalar>(1),
          cutlass::from_real<ElementScalar>(0),
          true,
          detail::Iterations(iterations)
        );

        if (!passed) {
          return false;
        }
      }
    }
  }


  // if we do support batched GEMM, just run it once
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{problem_size_m[0], problem_size_n[0], problem_size_k[0], /* l */ 4};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(1),
      cutlass::from_real<ElementScalar>(0),
      true,
      detail::Iterations(iterations)
    );

    if (!passed) {
      return false;
    }
  }

  return passed;
}


} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

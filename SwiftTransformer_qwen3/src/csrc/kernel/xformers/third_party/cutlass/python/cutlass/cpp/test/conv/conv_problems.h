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
/* \file
   \brief Bind convolution problems to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>


#include "unit/conv/device/conv2d_problems.h"
#include "cutlass/conv/conv2d_problem_size.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<cutlass::conv::Conv2dProblemSize>);

void bind_conv_problem_size_test(py::module &m) {
    
    py::bind_vector<std::vector<cutlass::conv::Conv2dProblemSize>>(m, "Conv2dProblemVector")
        .def("size", &std::vector<cutlass::conv::Conv2dProblemSize>::size);
    // Get Conv2d problem sizes
    py::class_<test::conv::device::TestbedConv2dProblemSizes>(m, "TestbedConv2dProblemSizes")
        .def(py::init<int>())
        .def_readonly("conv2d_default_sizes", &test::conv::device::TestbedConv2dProblemSizes::conv2d_default_sizes);
}

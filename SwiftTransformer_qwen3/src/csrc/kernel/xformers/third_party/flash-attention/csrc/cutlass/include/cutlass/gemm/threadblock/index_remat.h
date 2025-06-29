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
    \brief Helpers for rematerializing indices/dimensions in the thread hierarchy from special registers
*/

#pragma once

#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxX() {
  return threadIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxY() {
  return threadIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxZ() {
  return threadIdx.z;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX() {
  return blockIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY() {
  return blockIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxZ() {
  return blockIdx.z;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX() {
  return blockDim.x;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimY() {
  return blockDim.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimZ() {
  return blockDim.z;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass



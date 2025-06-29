#################################################################################################
#
# Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Low-level functionality tests for GEMM with F16 operands on SM80
"""

from functools import partial

import cutlass
import logging
import unittest

from cutlass.backend.test.utils import LayoutCombination, add_test_gemm
from cutlass.backend.utils.device import device_cc


cutlass.set_log_level(logging.WARNING)
cc = 80

@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
class GemmF16Sm80(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


@unittest.skipIf(device_cc() < cc, 'Device compute capability is insufficient for SM80 tests.')
class GemmF16Sm80StreamK(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass

add_test_specialized = partial(add_test_gemm, element=cutlass.DataType.f16, cc=cc, cluster_shape=[1, 1, 1])

# Tests using TensorOp
add_test_tensorop = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp)

add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NTN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.NTT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TTN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TTT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[ 64, 128, 32], warp_count=[1, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128,  64, 32], warp_count=[2, 1, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[ 64,  64, 64], warp_count=[1, 1, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[4, 4, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f32, threadblock_shape=[ 64,  64, 64], warp_count=[1, 1, 1], stages=5)
add_test_tensorop(cls=GemmF16Sm80, layouts=LayoutCombination.TNT, alignments=[2, 2, 2], element_output=cutlass.DataType.f16,
                  element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)

# Tests using SIMT
add_test_simt = partial(add_test_specialized, opclass=cutlass.OpcodeClass.Simt)

add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.NNN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16,
              element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.TNN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16,
              element_accumulator=cutlass.DataType.f32, threadblock_shape=[ 64, 128, 8], warp_count=[1, 2, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.NTN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16,
              element_accumulator=cutlass.DataType.f32, threadblock_shape=[128,  64, 8], warp_count=[2, 1, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.TTN, alignments=[1, 1, 1], element_output=cutlass.DataType.f16,
              element_accumulator=cutlass.DataType.f32, threadblock_shape=[ 64,  64, 8], warp_count=[1, 1, 1], stages=2)
add_test_simt(cls=GemmF16Sm80, layouts=LayoutCombination.NNT, alignments=[1, 1, 1], element_output=cutlass.DataType.f16,
              element_accumulator=cutlass.DataType.f16, threadblock_shape=[128, 128, 8], warp_count=[2, 2, 1], stages=2)

# Stream K tests
add_test_streamk = partial(add_test_specialized, opclass=cutlass.OpcodeClass.TensorOp, swizzle=cutlass.swizzle.ThreadblockSwizzleStreamK)
add_test_streamk(cls=GemmF16Sm80StreamK, layouts=LayoutCombination.NNN, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                 element_accumulator=cutlass.DataType.f32, threadblock_shape=[128, 128, 32], warp_count=[2, 2, 1], stages=3)
add_test_streamk(cls=GemmF16Sm80StreamK, layouts=LayoutCombination.TNT, alignments=[8, 8, 8], element_output=cutlass.DataType.f16,
                 element_accumulator=cutlass.DataType.f32, threadblock_shape=[ 64,  64, 64], warp_count=[1, 1, 1], stages=5)

if __name__ == '__main__':
    unittest.main()

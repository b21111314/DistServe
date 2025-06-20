#################################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cutlass
import cutlass_bindings

from cutlass import EpilogueScheduleSuffixes, KernelScheduleSuffixes
from cutlass.utils.datatypes import binding_opclass, binding_type
from cutlass.backend import library
from cutlass.backend.test.gemm_testbed import test_all_gemm
from cutlass.backend.utils.software import SubstituteTemplate


class Layout:
    """
    Utility class to map transpose and non-transpose terminology to row- and column-major terminology
    """

    T = cutlass_bindings.RowMajor
    N = cutlass_bindings.ColumnMajor


class LayoutCombination:
    """
    Utility class defining all combinations of row- and column-major layouts for operands to a GEMMs
    """

    NNN = (Layout.N, Layout.N, Layout.N)
    NNT = (Layout.N, Layout.N, Layout.T)
    NTN = (Layout.N, Layout.T, Layout.N)
    NTT = (Layout.N, Layout.T, Layout.T)
    TNN = (Layout.T, Layout.N, Layout.N)
    TNT = (Layout.T, Layout.N, Layout.T)
    TTN = (Layout.T, Layout.T, Layout.N)
    TTT = (Layout.T, Layout.T, Layout.T)


def get_name(
    layouts,
    alignments,
    element_output,
    element_accumulator,
    element_epilogue,
    cluster_shape,
    threadblock_shape,
    stages,
    element_a,
    element_b,
    arch,
    opclass,
    kernel_schedule=None,
    epilogue_schedule=None,
    suffix="",
):
    """
    Generates a procedural name for a test case.

    :param layouts: indexable container of layouts of A, B, and C operands
    :param alignments: indexable container of alignments of A, B, and C operands
    :param element_output: data type of the output element
    :param element_accumulator: data type used in accumulation
    :param element_epilogue: data type used in computing the epilogue
    :param cluster_shape: indexable container of dimensions of threadblock cluster to be launched
    :param threadblock_shape: indexable container of dimensions of threadblock tiles
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param element_a: data type of operand A
    :param element_b: data type of operand B
    :param arch: compute capability of kernel being generated
    :type arch: int
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass_bindings.OpClass
    :param kernel_schedule: kernel_schedule type
    :type kernel_schedule: cutlass.KernelScheduleType
    :param epilogue_schedule: epilogue_schedule type
    :type epilogue_schedule: cutlass.EpilogueScheduleType
    :param suffix: additional string to add to the suffix of the name
    :type suffix: str

    :return: str
    """
    name_format = "test_SM${arch}_Device_Gemm_${eA}${lA}_${eB}${lB}_${eC}${lC}_${opclass}_${acc}_${tbM}x${tbN}x${tbK}_${cM}x${cN}x${cK}_${stages}_align${aA}-${aB}-${aC}${k}${e}${suffix}"
    return SubstituteTemplate(
        name_format,
        {
            "arch": str(arch),
            "eA": library.DataTypeNames[binding_type(element_a)],
            "eB": library.DataTypeNames[binding_type(element_b)],
            "eC": library.DataTypeNames[binding_type(element_output)],
            "lA": library.ShortLayoutTypeNames[layouts[0]],
            "lB": library.ShortLayoutTypeNames[layouts[1]],
            "lC": library.ShortLayoutTypeNames[layouts[2]],
            "opclass": library.OpcodeClassNames[binding_opclass(opclass)],
            "acc": library.DataTypeNames[binding_type(element_accumulator)],
            "cM": str(cluster_shape[0]),
            "cN": str(cluster_shape[1]),
            "cK": str(cluster_shape[2]),
            "tbM": str(threadblock_shape[0]),
            "tbN": str(threadblock_shape[1]),
            "tbK": str(threadblock_shape[2]),
            "stages": str(stages) if stages is not None else "auto",
            "aA": str(alignments[0]),
            "aB": str(alignments[1]),
            "aC": str(alignments[2]),
            "k": "" if kernel_schedule is None else KernelScheduleSuffixes[kernel_schedule],
            "e": "" if epilogue_schedule is None else EpilogueScheduleSuffixes[epilogue_schedule],
            "suffix": "" if suffix is None else suffix,
        },
    )

def get_name_conv2d(
    arch,
    conv_kind,
    element,
    element_accumulator,
    element_output,
    opclass,
    threadblock_shape,
    warp_count,
    instruction_shape,
    stages,
    iterator_algorithm,
    swizzle,
    split_k_mode,
    split_k_slices,
    activation
):
    """
    Generates a procedural name for a test case for conv2d
    
    :param arch: compute capability of kernel being generated
    :type arch: int
    :param conv_kind: the convolution type (i.e. fprop, dgrad, wgrad)
    :type conv_kind: str
    :param iterator_algorithm: the iterator algorithm applied
    :type iterator_algorithm: cutlass_bindings.conv.IteratorAlgorithm
    :param element_a: data type of operand A
    :param element_b: data type of operand B
    :param element_c: data type of operand C
    :param element_accumulator: data type used in accumulation
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass_bindings.OpClass
    :param threadblock_shape: indexable container of dimensions of threadblock tiles
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param stride_support: stride support of dgrad
    :param alignment: int
    :type alignment: int
    
    :return: str
    """
    if iterator_algorithm is None:
        iterator_algorithm = "AUTO"
    if swizzle is None:
        swizzle = 1
    name_format = "test_SM${arch}_Device_Conv2d_${conv_kind}_${iter_alg}_ImplicitGemm_${eA}nhwc_${eB}nhwc_${eC}nhwc_${opclass}_${acc}_${tbM}x${tbN}x${tbK}_${wM}x${wN}x${wK}_${IM}${IN}${IK}_stage${stages}_swizzle${swizzle}_${split_k_mode}${split_k_slices}_${activation}"
    
    return SubstituteTemplate(
        name_format,
        {
            "arch": str(arch),
            "conv_kind": conv_kind,
            "iter_alg": iterator_algorithm,
            "eA": library.DataTypeNames[binding_type(element)],
            "eB": library.DataTypeNames[binding_type(element)],
            "eC": library.DataTypeNames[binding_type(element_output)],
            "opclass": opclass,
            "acc": library.DataTypeNames[binding_type(element_accumulator)],
            "tbM": str(threadblock_shape[0]),
            "tbN": str(threadblock_shape[1]),
            "tbK": str(threadblock_shape[2]),
            "wM": str(threadblock_shape[0] // warp_count[0]),
            "wN": str(threadblock_shape[1] // warp_count[1]),
            "wK": str(threadblock_shape[2] // warp_count[2]),
            "IM": str(instruction_shape[0]),
            "IN": str(instruction_shape[1]),
            "IK": str(instruction_shape[2]),
            "stages": str(stages),
            "swizzle": str(swizzle),
            "split_k_mode": split_k_mode,
            "split_k_slices": str(split_k_slices),
            "activation": activation
        }
    )
    

def add_test_gemm(
    cls=None,
    cc=None,
    element=None,
    layouts=None,
    alignments=None,
    element_output=None,
    element_accumulator=None,
    cluster_shape=None,
    threadblock_shape=None,
    warp_count=None,
    stages=None,
    opclass=None,
    swizzle=None,
    kernel_schedule=None,
    epilogue_schedule=None,
    compilation_modes=['nvcc', 'nvrtc']):
    """
    Create test-running functions with the given specification and set it as a method of ``cls``.

    :param cls: class to which the generated method will be added
    :type cls: type
    :param cc: compute capability to compile for
    :type cc: int
    :param element: data type of A and B operands
    :type element: cutlass.DataType.f16
    :param layouts: layouts of A, B, and C operands
    :type layouts: list or tuple
    :param alignments: alingments of A, B, and C operands
    :type alignments: list or tuple
    :param element_output: data type of the output element
    :type element_output: cutlass.DataType
    :param element_accumulator: data type used in accumulation
    :type element_accumulator: cutlass.DataType
    :param cluster_shape: dimensions of clusters
    :type cluster_shape: list or tuple
    :param threadblock_shape: dimensions of threadblock tiles
    :type threadblock_shape: list or tuple
    :param warp_count: warps to be launched per threadblock dimension
    :type warp_count: list or tuple
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass.OpClass
    :param swizzle: threadblock swizzling functor
    :param kernel_schedule: kernel schedule to use
    :type kernel_schedule: cutlass.KernelScheduleType
    :param epilogue_schedule: epilogue schedule to use
    :type epilogue_schedule: cutlass.EpilogueScheduleType
    :param compilation_modes: list of compilers to used in testing the kernel (options: 'nvrtc', 'nvcc')
    :type compilation_modes: list
    """

    for compilation_mode in compilation_modes:
        def run(self):
            """
            Dynamically-generated function that constructs a GEMM operation and verifies it against
            multiple test cases.
            """
            element_A = element
            element_B = element
            layout_A, layout_B, layout_C = layouts
            alignment_A, alignment_B, alignment_C = alignments

            plan = cutlass.op.Gemm(element_A=element_A, element_B=element_B,
                                element_C=element_output, element_D=element_output,
                                layout_A=layout_A, layout_B=layout_B, layout_C=layout_C,
                                element_accumulator=element_accumulator,
                                kernel_cc=cc)

            plan.opclass = opclass
            if swizzle is not None:
                plan.swizzling_functor = swizzle
            td = plan.tile_descriptions()[0]
            td.threadblock_shape = threadblock_shape
            td.stages = stages
            if warp_count is not None:
                td.warp_count = warp_count
            td.cluster_shape = cluster_shape
            op = plan.construct(tile_description=td, alignment_A=alignment_A, alignment_B=alignment_B, alignment_C=alignment_C)
            self.assertTrue(test_all_gemm(op, 'universal', compilation_mode=compilation_mode))

        element_epilogue = element_accumulator
        name = get_name(
            layouts=layouts, alignments=alignments, element_output=element_output, element_accumulator=element_accumulator,
            element_epilogue=element_epilogue, cluster_shape=cluster_shape, threadblock_shape=threadblock_shape,
            stages=stages, element_a=element, element_b=element, arch=cc, opclass=opclass,
            kernel_schedule=kernel_schedule, epilogue_schedule=epilogue_schedule, suffix=f'_{compilation_mode}')

        setattr(cls, name, run)

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

import ctypes

import cutlass_bindings
from cutlass import (
    DataType,
    KernelScheduleType
)
from cutlass.backend.library import DataTypeSizeBytes


class GemmCoord_(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int)
    ]

    def __init__(self, gemm_coord) -> None:
        for field_name, _ in self._fields_:
            setattr(self, field_name, getattr(gemm_coord, field_name)())


class GemmCoordBatched_(ctypes.Structure):
    """
    Wrapper around a GemmCoord that also contains batch count. This is used for encoding
    batched GEMM inputs to CUTLASS 3 GEMMs.
    """

    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
        ("batch_count", ctypes.c_int)
    ]

    def __init__(self, gemm_coord, batch_count) -> None:
        for field_name, _ in self._fields_[:-1]:
            setattr(self, field_name, getattr(gemm_coord, field_name)())
        setattr(self, "batch_count", batch_count)


class MatrixCoord_(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_int),
        ("column", ctypes.c_int)
    ]


class dim3_(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int)
    ]


class StrideBatched_(ctypes.Structure):
    """
    CUTLASS 3.0 strides for operands contain one static dimension and two variable dimensions. The
    variable dimensions represent the stride along non-unit-stride dimension of the row/column major
    layout, and the batch stride. This structure encodes the two variable dimensions.
    """
    _fields_ = [
        ("major_stride", ctypes.c_int64),
        ("batch_stride", ctypes.c_int64)
    ]


dtype2ctype = {
    cutlass_bindings.float16: ctypes.c_uint16,
    cutlass_bindings.float32: ctypes.c_float,
    cutlass_bindings.float64: ctypes.c_double,
    cutlass_bindings.int32: ctypes.c_int32,
}


class GenericMainloopArguments3x_(ctypes.Structure):
    """
    Structure representing the superset of possible mainloop arguments.
    This structure should not be passed to kernels directly, but, rather,
    be used as an input to one of the more specific schedule arguments, which
    will each select those arguments relevant to the particular schedule.
    """
    _fields_ = [
        ("ptr_A", ctypes.c_void_p),
        ("stride_A", StrideBatched_),
        ("ptr_B", ctypes.c_void_p),
        ("stride_B", StrideBatched_),
        ("mma_promotion_interval", ctypes.c_int)
    ]


def get_mainloop_arguments_3x(
    kernel_schedule: KernelScheduleType,
    element_A,
    element_B,
    alignment_A: int,
    alignment_B: int) -> ctypes.Structure:
    """
    Returns the ctypes structure to be used for the 3.x kernel's mainloop parameters.

    :param kernel_schedule: type of kernel schedule to be used in the mainloop
    :type kerel_schedule: cutlass.KernelScheduleType
    :param element_A: data type of operand A
    :param element_B: data type of operand B
    :param alignment_A: alignment of operand A
    :type alignment_A: int
    :param alignment_B: alignment of operand B
    :type alignment_B: int

    :returns: ctypes structure to be used for the 3.x kernel's mainloop parameters
    :rtype: ctypes.Structure
    """
    class _MainloopArgumentsTma(ctypes.Structure):
        _fields_ = [
            ("ptr_A", ctypes.c_void_p),
            ("stride_A", StrideBatched_),
            ("ptr_B", ctypes.c_void_p),
            ("stride_B", StrideBatched_),
            ("mma_promotion_interval", ctypes.c_int)
        ]

        @staticmethod
        def from_generic_mainloop_args(args: GenericMainloopArguments3x_):
            return _MainloopArgumentsTma(
                args.ptr_A, args.stride_A, args.ptr_B, args.stride_B,
                args.mma_promotion_interval
            )

    class _MainloopArgumentsMultistage(ctypes.Structure):
        _fields_ = [
            ("ptr_A", ctypes.c_void_p),
            ("stride_A", StrideBatched_),
            ("ptr_B", ctypes.c_void_p),
            ("stride_B", StrideBatched_),
        ]

        @staticmethod
        def from_generic_mainloop_args(args: GenericMainloopArguments3x_):
            return _MainloopArgumentsMultistage(
                args.ptr_A, args.stride_A, args.ptr_B, args.stride_B,
            )

    tma_alignment_bytes = 16
    is_tma_aligned_A = ((DataTypeSizeBytes[element_A] * alignment_A) % tma_alignment_bytes) == 0
    is_tma_aligned_B = ((DataTypeSizeBytes[element_B] * alignment_B) % tma_alignment_bytes) == 0
    is_tma_aligned = is_tma_aligned_A and is_tma_aligned_B

    if kernel_schedule == KernelScheduleType.Multistage:
        return _MainloopArgumentsMultistage
    elif kernel_schedule == KernelScheduleType.ScheduleAuto:
        if is_tma_aligned:
            return _MainloopArgumentsTma
        else:
            return _MainloopArgumentsMultistage
    else:
        if is_tma_aligned:
            return _MainloopArgumentsTma
        else:
            raise Exception(f"Specified a kernel schedule using TMA ({kernel_schedule}), but "
                            "the provided data types and alignments are not properly aligned for "
                            "using TMA.")


def get_gemm_arguments_3x(mainloop_arguments, epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _EpilogueArguments(ctypes.Structure):
        _fields_ = [
            ("epilogue", _EpilogueOutputOpParams),
            ("ptr_C", ctypes.c_void_p),
            ("stride_C", StrideBatched_),
            ("ptr_D", ctypes.c_void_p),
            ("stride_D", StrideBatched_),
        ]

    class _HardwareInfo(ctypes.Structure):
        _fields_ = [
            ("device_id", ctypes.c_int),
            ("sm_count", ctypes.c_int)
        ]

    class _GemmArguments(ctypes.Structure):
        _fields_ = [
            ("mode", ctypes.c_int),
            ("problem_size", GemmCoordBatched_),
            ("mainloop", mainloop_arguments),
            ("epilogue", _EpilogueArguments),
            ("hw_info", _HardwareInfo),
            ("splits", ctypes.c_int)
        ]

    return _GemmArguments, _EpilogueArguments, _EpilogueOutputOpParams, _HardwareInfo


def get_gemm_arguments(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _GemmArguments(ctypes.Structure):
        _fields_ = [
            # Arguments from UniversalArgumentsBase
            ("mode", ctypes.c_int),
            ("problem_size", GemmCoord_),
            ("batch_count", ctypes.c_int),
            ("batch_stride_D", ctypes.c_longlong),
            # Remaining arguments
            ("epilogue", _EpilogueOutputOpParams),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("batch_stride_A", ctypes.c_longlong),
            ("batch_stride_B", ctypes.c_longlong),
            ("batch_stride_C", ctypes.c_longlong),
            ("stride_a", ctypes.c_longlong),
            ("stride_b", ctypes.c_longlong),
            ("stride_c", ctypes.c_longlong),
            ("stride_d", ctypes.c_longlong),
            ("lda", ctypes.c_longlong),
            ("ldb", ctypes.c_longlong),
            ("ldc", ctypes.c_longlong),
            ("ldd", ctypes.c_longlong),
            ("ptr_gather_A_indices", ctypes.c_void_p),
            ("ptr_gather_B_indices", ctypes.c_void_p),
            ("ptr_scatter_D_indices", ctypes.c_void_p)
        ]

    return _GemmArguments, _EpilogueOutputOpParams


def get_gemm_arguments_streamk(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _GemmArguments(ctypes.Structure):
        _fields_ = [
            ("mode", ctypes.c_int),
            ("problem_size", GemmCoord_),
            ("batch_count", ctypes.c_int),
            ("epilogue", _EpilogueOutputOpParams),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("batch_stride_A", ctypes.c_longlong),
            ("batch_stride_B", ctypes.c_longlong),
            ("batch_stride_C", ctypes.c_longlong),
            ("batch_stride_D", ctypes.c_longlong),
            ("stride_a", ctypes.c_longlong),
            ("stride_b", ctypes.c_longlong),
            ("stride_c", ctypes.c_longlong),
            ("stride_d", ctypes.c_longlong),
            ("lda", ctypes.c_longlong),
            ("ldb", ctypes.c_longlong),
            ("ldc", ctypes.c_longlong),
            ("ldd", ctypes.c_longlong),
            ("avail_sms", ctypes.c_int)
        ]

    return _GemmArguments, _EpilogueOutputOpParams


###########################################################################################
# GEMM Grouped
###########################################################################################


def get_gemm_grouped_arguments(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _GEMMGroupedArguments(ctypes.Structure):
        _fields_ = [
            ("problem_sizes", ctypes.c_void_p),
            ("problem_count", ctypes.c_int),
            ("threadblock_count", ctypes.c_int),
            ("output_op", _EpilogueOutputOpParams),
            ("ptr_A", ctypes.c_void_p),
            ("ptr_B", ctypes.c_void_p),
            ("ptr_C", ctypes.c_void_p),
            ("ptr_D", ctypes.c_void_p),
            ("lda", ctypes.c_void_p),
            ("ldb", ctypes.c_void_p),
            ("ldc", ctypes.c_void_p),
            ("ldd", ctypes.c_void_p),
            ("host_problem_sizes", ctypes.c_void_p)
        ]

    return _GEMMGroupedArguments, _EpilogueOutputOpParams


############################################################################################
# Convolution2D
############################################################################################


class Conv2DProblemSize(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_int),
        ("H", ctypes.c_int),
        ("W", ctypes.c_int),
        ("C", ctypes.c_int),
        ("P", ctypes.c_int),
        ("Q", ctypes.c_int),
        ("K", ctypes.c_int),
        ("R", ctypes.c_int),
        ("S", ctypes.c_int),
        ("pad_h", ctypes.c_int),
        ("pad_w", ctypes.c_int),
        ("stride_h", ctypes.c_int),
        ("stride_w", ctypes.c_int),
        ("dilation_h", ctypes.c_int),
        ("dilation_w", ctypes.c_int),
        ("mode", ctypes.c_int),  # kCrossCorrelation: 0, kConvolution: 1
        ("split_k_slices", ctypes.c_int),
        ("groups", ctypes.c_int)
    ]

    def __init__(self, problem_size) -> None:
        for field_name, _ in self._fields_:
            setattr(self, field_name, getattr(problem_size, field_name))


class Layout4D(ctypes.Structure):
    _fields_ = [("stride", ctypes.c_int * 3)]

    def __init__(self, tensor_ref):
        stride = tensor_ref.stride()
        setattr(self, "stride", (stride.at(0), stride.at(1), stride.at(2)))


class TensorRef_(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("layout", Layout4D)
    ]

    def __init__(self, tensor_ref):
        setattr(self, "ptr", tensor_ref.data())
        setattr(self, "layout", Layout4D(tensor_ref.layout()))


class TensorRef2D_(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("stride", ctypes.c_int)
    ]


def get_conv2d_arguments(epilogue_functor):
    _EpilogueOutputOpParams = epilogue_functor.epilogue_type

    class _Conv2dArguments(ctypes.Structure):
        _fields_ = [
            ("problem_size", Conv2DProblemSize),
            ("ref_A", TensorRef_),
            ("ref_B", TensorRef_),
            ("ref_C", TensorRef_),
            ("ref_D", TensorRef_),
            ("output_op", _EpilogueOutputOpParams),
            ("split_k_mode", ctypes.c_int)
        ]

    return _Conv2dArguments, _EpilogueOutputOpParams


############################################################################################
# Reduction
############################################################################################


def get_reduction_params(epilogue_functor):
    _EpilogueOutputParams = epilogue_functor.epilogue_type

    class _ReductionParams(ctypes.Structure):
        _fields_ = [
            ("problem_size", MatrixCoord_),
            ("partitions", ctypes.c_int),
            ("partition_stride", ctypes.c_longlong),
            ("workspace", TensorRef2D_),
            ("destination", TensorRef2D_),
            ("source", TensorRef2D_),
            ("output_op", _EpilogueOutputParams),
        ]

    return _ReductionParams, _EpilogueOutputParams

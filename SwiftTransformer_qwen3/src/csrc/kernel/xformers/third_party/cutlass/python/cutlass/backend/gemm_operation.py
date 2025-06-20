################################################################################
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
################################################################################

import copy
import ctypes
import enum

from cuda import cuda, cudart
import cutlass_bindings
import numpy as np
import rmm

from cutlass import (
    EpilogueScheduleSuffixes,
    EpilogueScheduleTag,
    EpilogueScheduleType,
    KernelScheduleSuffixes,
    KernelScheduleTag,
    KernelScheduleType,
    TileSchedulerSuffixes,
    TileSchedulerTag,
    TileSchedulerType
)
from cutlass.backend.arguments import ArgumentBase
from cutlass.backend.c_types import (
    GemmCoord_,
    GemmCoordBatched_,
    GenericMainloopArguments3x_,
    StrideBatched_,
    dim3_,
    get_gemm_arguments,
    get_gemm_arguments_3x,
    get_gemm_arguments_streamk,
    get_gemm_grouped_arguments,
    get_mainloop_arguments_3x
)
from cutlass.backend.library import (
    ApiVersion,
    EmissionType,
    ComplexTransformTag,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    GemmKind,
    GemmKindNames,
    LayoutTag,
    MathOperation,
    MathOperationTag,
    OpcodeClassNames,
    OpcodeClassTag,
    OperationKind,
    SchedulerMode,
    SchedulerModeTag,
    ShortComplexLayoutNames,
    ShortDataTypeNames,
    ShortLayoutTypeNames,
    TensorDescription,
    TileDescription,
    api_version,
    enum_auto,
    get_complex_from_real,
)
from cutlass.backend.memory_manager import device_mem_alloc, todevice
from cutlass.backend.operation import ExecutableOperation, LaunchConfiguration
from cutlass.backend.tensor_ref import TensorRef
from cutlass.backend.type_hint import GemmOperation, Tensor
from cutlass.backend.utils.software import (
    CheckPackages,
    SubstituteTemplate,
    device_sm_count,
)

if CheckPackages().check_torch():
    import torch


################################################################################
#
# Data structure modeling a GEMM operation
#
################################################################################


def transpose_layout(layout: cutlass_bindings.layout):
    if layout == cutlass_bindings.ColumnMajor:
        return cutlass_bindings.RowMajor
    elif layout == cutlass_bindings.RowMajor:
        return cutlass_bindings.ColumnMajor
    else:
        raise ValueError("unsupported Layout {}".format(layout))


class GemmArguments2x(ArgumentBase):
    """
    Argument wrapper for GEMM in CUTLASS 2. It encodes problem information and
    user-provide tensors into the kernel's argument

    :param operation: the GEMM operation to take the argument
    :type operation: :class:`cutlass.backend.GemmOperationUniversal` |
     :class:`cutlass.backend.GemmOperationGrouped`

    :param problem_size: GEMM problem size gemm(M, N, K)
    :type operation: :class:`cutlass_bindings.gemm.GemmCoord`

    :param A: tensor A
    :type A: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param B: tensor B
    :type B: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param C: tensor C
    :type C: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param D: tensor D
    :type D: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param gemm_mode: GEMM mode
    :type gemm_mode: :class:`cutlass_bindings.gemm.Mode`

    :param output_op: output operator, optional
    :type output_op: :class:`cutlass.backend.LinearCombinationFunctorArguments`
    """

    def __init__(
        self, operation: "GemmOperation", problem_size: "cutlass_bindings.gemm.GemmCoord",
        A: "Tensor", B: "Tensor", C: "Tensor", D: "Tensor",
        gemm_mode: "cutlass_bindings.gemm.Mode" = cutlass_bindings.gemm.Mode.Gemm, **kwargs):
        self.operation = operation

        self.layout_A: cutlass_bindings.layout = operation.A.layout
        self.layout_B: cutlass_bindings.layout = operation.B.layout
        self.layout_C: cutlass_bindings.layout = operation.C.layout

        self.element_A = operation.A.element
        self.element_B = operation.B.element
        self.element_C = operation.C.element

        if (operation.C.layout in
            [cutlass_bindings.RowMajorInterleaved32, cutlass_bindings.ColumnMajorInterleaved32]):
            # reorder tensor B for interleaved layout output
            B = self.reorder_tensor_B(B, problem_size)

        super().__init__(A, B, C, D, **kwargs)

        if operation.switched:
            self.problem_size = cutlass_bindings.gemm.GemmCoord(
                problem_size.n(), problem_size.m(), problem_size.k())
            self.ptr_A, self.ptr_B = self.ptr_B, self.ptr_A
        else:
            self.problem_size = cutlass_bindings.gemm.GemmCoord(
                problem_size.m(), problem_size.n(), problem_size.k())

        # if the number of elements in C = problem_size.n
        # C is treated as the bias
        if hasattr(self, "tensor_c_numel"):
            if self.tensor_c_numel == self.problem_size.n() and self.problem_size.m() != 1:
                self.bias = True

        # get the leading dimension
        self.lda = operation.A.layout.packed(self.problem_size.mk()).stride()
        self.ldb = operation.B.layout.packed(self.problem_size.kn()).stride()
        self.ldc = operation.C.layout.packed(self.problem_size.mn()).stride()
        self.ldd = self.ldc

        # stride 0 trick
        if self.bias:
            self.ldc = 0

        if "output_op" in kwargs.keys() and gemm_mode != cutlass_bindings.gemm.Mode.GemmSplitKParallel:
            self.output_op = kwargs["output_op"]
        else:
            self.output_op = self.operation.epilogue_type(1.0, 0.0)

        # get number of slices on k dimension
        self.gemm_mode = gemm_mode
        if gemm_mode in [cutlass_bindings.gemm.Mode.Gemm, cutlass_bindings.gemm.Mode.GemmSplitKParallel]:
            if "split_k_slices" in kwargs.keys():
                self.batch_count = kwargs["split_k_slices"]
            else:
                self.batch_count = 1
            self.split_k_slices = self.batch_count

        if gemm_mode in [cutlass_bindings.gemm.Mode.Batched, cutlass_bindings.gemm.Mode.Array]:
            if "batch" in kwargs.keys():
                self.batch_count = kwargs["batch"]
            else:
                self.batch_count = 1

        if "batch_strides" in kwargs:
            self.batched_stride_A = kwargs["batch_strides"]["A"]
            self.batched_stride_B = kwargs["batch_strides"]["B"]
            self.batched_stride_C = kwargs["batch_strides"]["C"]
            self.batched_stride_D = kwargs["batch_strides"]["D"]
        else:
            self.batched_stride_A = self.problem_size.m() * self.problem_size.k()
            self.batched_stride_B = self.problem_size.n() * self.problem_size.k()
            self.batched_stride_C = self.problem_size.m() * self.problem_size.n()
            self.batched_stride_D = self.problem_size.m() * self.problem_size.n()

        if self.bias:
            self.batched_stride_C = self.problem_size.n()

        # support GEMM Mode Array
        if gemm_mode == cutlass_bindings.gemm.Mode.Array:
            self.ptr_A_array = []
            self.ptr_B_array = []
            self.ptr_C_array = []
            self.ptr_D_array = []

            ptr_A_addr = int(self.ptr_A)
            ptr_B_addr = int(self.ptr_B)
            ptr_C_addr = int(self.ptr_C)
            ptr_D_addr = int(self.ptr_D)

            stride_A = self.batched_stride_A * DataTypeSize[self.element_A] // 8
            stride_B = self.batched_stride_B * DataTypeSize[self.element_B] // 8
            stride_C = self.batched_stride_C * DataTypeSize[self.element_C] // 8
            stride_D = self.batched_stride_D * DataTypeSize[self.element_C] // 8
            for _ in range(self.batch_count):
                self.ptr_A_array.append(ptr_A_addr)
                self.ptr_B_array.append(ptr_B_addr)
                self.ptr_C_array.append(ptr_C_addr)
                self.ptr_D_array.append(ptr_D_addr)

                ptr_A_addr += stride_A
                ptr_B_addr += stride_B
                ptr_C_addr += stride_C
                ptr_D_addr += stride_D

            self.ptr_A_array_buffer = todevice(self.ptr_A_array, dtype=np.int64)
            self.ptr_B_array_buffer = todevice(self.ptr_B_array, dtype=np.int64)
            self.ptr_C_array_buffer = todevice(self.ptr_C_array, dtype=np.int64)
            self.ptr_D_array_buffer = todevice(self.ptr_D_array, dtype=np.int64)

        if isinstance(self.operation, GemmOperationUniversal):
            self.initialize()

    def reorder_tensor_B(self, tensor_B: "np.ndarray",
        problem_size: "cutlass_bindings.gemm.GemmCoord"):
        """
        Reorder tensor_B for interleaved layout

        :param tensor_B: input tensor B
        :type tensor_B: numpy.ndarray
        :param problem_size: GEMM problem size
        :type problem_size: :class:`cutlass_bindings.gemm.GemmCoord`

        :return: reordered tensor B
        :rtype: numpy.ndarray
        """
        reordered_tensor_B = np.empty_like(tensor_B)
        tensor_ref_B = self.get_tensor_ref(
            tensor_B, self.element_B, self.layout_B, problem_size, "b"
        )
        reordered_tensor_ref_B = self.get_tensor_ref(
            reordered_tensor_B, self.element_B, self.layout_B, problem_size, "b"
        )
        cutlass_bindings.gemm.host.reorder_column(
            tensor_ref_B, reordered_tensor_ref_B, problem_size)
        return reordered_tensor_B

    def get_tensor_ref(
        self, tensor, dtype, tensor_layout, problem_size, operand):
        if operand == "a":
            tensor_coord = problem_size.mk()
        elif operand == "b":
            tensor_coord = problem_size.kn()
        elif operand in ["c", "d"]:
            tensor_coord = problem_size.mn()
        else:
            raise ValueError("unknown operand: " + operand)

        layout = tensor_layout.packed(tensor_coord)

        return TensorRef(tensor, dtype, layout).tensor_ref

    def get_arguments(self):
        problem_size_ = GemmCoord_(self.problem_size)
        grid_tiled_shape_ = GemmCoord_(
            cutlass_bindings.gemm.GemmCoord(
                self.grid_tiled_shape.x,
                self.grid_tiled_shape.y,
                self.grid_tiled_shape.z
            )
        )
        if self.gemm_mode == cutlass_bindings.gemm.Mode.Array:
            arguments = self.operation.argument_type(
                # Arguments from UniversalArgumentsBase
                self.gemm_mode,
                problem_size_,
                self.batch_count,
                0,
                # Remaining arguments
                self.output_op,
                int(self.ptr_A_array_buffer.ptr),
                int(self.ptr_B_array_buffer.ptr),
                int(self.ptr_C_array_buffer.ptr),
                int(self.ptr_D_array_buffer.ptr),
                0, 0, 0,
                self.lda, self.ldb, self.ldc, self.ldd,
                self.lda, self.ldb, self.ldc, self.ldd,
                0, 0, 0
            )
        else:
            arguments = self.operation.argument_type(
                # Arguments from UniversalArgumentsBase
                self.gemm_mode, problem_size_, self.batch_count, self.batched_stride_D,
                # Remaining arguments
                self.output_op,
                int(self.ptr_A),
                int(self.ptr_B),
                int(self.ptr_C),
                int(self.ptr_D),
                self.batched_stride_A,
                self.batched_stride_B,
                self.batched_stride_C,
                self.lda, self.ldb, self.ldc, self.ldd,
                self.lda, self.ldb, self.ldc, self.ldd,
                0, 0, 0
            )

        self.arguments = arguments, grid_tiled_shape_, self.gemm_k_size

    def initialize(self):
        # get launch configuration
        launch_config = self.operation.rt_module.plan(self)

        # get the host and evice workspace
        device_workspace_size = self.operation.rt_module.get_device_workspace_size(self)

        if device_workspace_size > 0:
            self.workspace_buffer = device_mem_alloc(device_workspace_size)
            workspace_ptr = self.workspace_buffer.ptr
            err, = cuda.cuMemsetD32(
                workspace_ptr, 0, device_workspace_size // 4)
        else:
            workspace_ptr = None

        device_workspace = 0
        if workspace_ptr is not None and self.gemm_mode == cutlass_bindings.gemm.Mode.GemmSplitKParallel:
            # in GEMM splik-K parallel, the D pointer is redirected
            # to the workspace
            self.ptr_D = cuda.CUdeviceptr(workspace_ptr)
        elif workspace_ptr is not None and self.gemm_mode == cutlass_bindings.gemm.Mode.Gemm:
            # in GEMM split-K serial
            device_workspace = workspace_ptr

        self.get_arguments()

        arguments, grid_tiled_shape, gemm_k_size = self.arguments
        res_arg = self.operation.rt_module.get_args(
            ctypes.byref(arguments), ctypes.c_void_p(int(device_workspace)))
        host_workspace = bytearray(res_arg.contents)

        device_workspace = None

        self.host_workspace = host_workspace
        self.device_workspace = device_workspace
        self.launch_config = launch_config


class GemmArguments2xStreamK(GemmArguments2x):
    """
    Argument wrapper for stream-K GEMMs in CUTLASS 2. It encodes problem information and
    user-provide tensors into the kernel's argument

    :param operation: the GEMM operation to take the argument
    :type operation: :class:`cutlass.backend.GemmOperationUniversal` |
     :class:`cutlass.backend.GemmOperationGrouped`

    :param problem_size: GEMM problem size gemm(M, N, K)
    :type operation: :class:`cutlass_bindings.gemm.GemmCoord`

    :param A: tensor A
    :type A: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param B: tensor B
    :type B: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param C: tensor C
    :type C: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param D: tensor D
    :type D: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param gemm_mode: GEMM mode
    :type gemm_mode: :class:`cutlass_bindings.gemm.Mode`

    :param output_op: output operator, optional
    :type output_op: :class:`cutlass.backend.LinearCombinationFunctorArguments`
    """

    def __init__(
        self, operation: "GemmOperation", problem_size: "cutlass_bindings.gemm.GemmCoord",
        A: "Tensor", B: "Tensor", C: "Tensor", D: "Tensor",
        gemm_mode: "cutlass_bindings.gemm.Mode" = cutlass_bindings.gemm.Mode.Gemm, **kwargs):
        if gemm_mode not in [cutlass_bindings.gemm.Mode.Gemm, cutlass_bindings.gemm.Mode.Batched]:
            raise Exception("Unsupporged GEMM mode {}.".format(gemm_mode))

        super().__init__(operation, problem_size, A, B, C, D, gemm_mode, **kwargs)

    def get_arguments(self):
        batch_stride_A = self.problem_size.m() * self.problem_size.k()
        batch_stride_B = self.problem_size.k() * self.problem_size.n()
        batch_stride_C = self.problem_size.m() * self.problem_size.n()
        batch_stride_D = self.problem_size.m() * self.problem_size.n()

        arguments = self.operation.argument_type(
            self.gemm_mode,
            GemmCoord_(self.problem_size),
            self.batch_count,
            self.output_op,
            int(self.ptr_A),
            int(self.ptr_B),
            int(self.ptr_C),
            int(self.ptr_D),
            batch_stride_A,
            batch_stride_B,
            batch_stride_C,
            batch_stride_D,
            self.lda, self.ldb, self.ldc, self.ldd,  # strides
            self.lda, self.ldb, self.ldc, self.ldd,
            -1,  # avail_sms
        )
        return arguments

    def initialize(self):
        # get the host and device workspace
        device_workspace_size = self.operation.rt_module.get_device_workspace_size(self)

        device_workspace_size = 10 << 20
        if device_workspace_size > 0:
            self.workspace_buffer = device_mem_alloc(device_workspace_size)
            workspace_ptr = self.workspace_buffer.ptr
            err, = cuda.cuMemsetD32(
                workspace_ptr, 0, device_workspace_size // 4)
        else:
            workspace_ptr = None

        device_workspace = 0
        if workspace_ptr is not None and self.gemm_mode == cutlass_bindings.gemm.Mode.GemmSplitKParallel:
            # in GEMM splik-K parallel, the D pointer is redirected
            # to the workspace
            self.ptr_D = cuda.CUdeviceptr(workspace_ptr)
        elif workspace_ptr is not None and self.gemm_mode == cutlass_bindings.gemm.Mode.Gemm:
            # in GEMM split-K serial
            device_workspace = workspace_ptr

        arguments = self.get_arguments()

        res_arg = self.operation.rt_module.get_args(
            ctypes.byref(arguments),
            ctypes.c_void_p(int(device_workspace)),
            device_sm_count(),
            self.operation.rt_module.occupancy
        )
        host_workspace = bytearray(res_arg.contents)

        grid = self.operation.rt_module.get_grid_shape(
            ctypes.byref(arguments),
            device_sm_count(),
            self.operation.rt_module.occupancy
        )

        device_workspace = None

        self.host_workspace = host_workspace
        self.device_workspace = device_workspace
        self.launch_config = LaunchConfiguration(
            [grid.m, grid.n, grid.k],
            [self.operation.rt_module.threads, 1, 1],
            self.operation.rt_module.shared_memory_capacity
        )


class GemmArguments3x(GemmArguments2x):
    """
    Argument wrapper for GEMM in CUTLASS 3. It encodes problem information and
    user-provide tensors into the kernel's argument

    :param operation: the GEMM operation to take the argument
    :type operation: :class:`cutlass.backend.GemmOperationUniversal` |
     :class:`cutlass.backend.GemmOperationGrouped`

    :param problem_size: GEMM problem size gemm(M, N, K)
    :type operation: :class:`cutlass_bindings.gemm.GemmCoord`

    :param A: tensor A
    :type A: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param B: tensor B
    :type B: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param C: tensor C
    :type C: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param D: tensor D
    :type D: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param gemm_mode: GEMM mode
    :type gemm_mode: :class:`cutlass_bindings.gemm.Mode`

    :param output_op: output operator, optional
    :type output_op: :class:`cutlass.backend.LinearCombinationFunctorArguments`
    """

    def __init__(
        self, operation: "GemmOperation", problem_size: "cutlass_bindings.gemm.GemmCoord",
        A: "Tensor", B: "Tensor", C: "Tensor", D: "Tensor",
        gemm_mode: "cutlass_bindings.gemm.Mode" = cutlass_bindings.gemm.Mode.Gemm, **kwargs):
        if gemm_mode not in [cutlass_bindings.gemm.Mode.Gemm, cutlass_bindings.gemm.Mode.Batched]:
            raise Exception("Unsupporged GEMM mode {}.".format(gemm_mode))

        super().__init__(operation, problem_size, A, B, C, D, gemm_mode, **kwargs)

    def get_arguments(self):
        problem_size_ = GemmCoordBatched_(self.problem_size, self.batch_count)

        if self.batch_count > 1:
            bsA = self.batched_stride_A
            bsB = self.batched_stride_B
            bsC = self.batched_stride_C
            bsD = self.batched_stride_D
        else:
            bsA = 0
            bsB = 0
            bsC = 0
            bsD = 0
        stride_A = StrideBatched_(self.lda, bsA)
        stride_B = StrideBatched_(self.ldb, bsB)
        stride_C = StrideBatched_(self.ldc, bsC)
        stride_D = StrideBatched_(self.ldd, bsD)

        # Superset of potential mainloop arguments
        generic_args = GenericMainloopArguments3x_(
            int(self.ptr_A),
            stride_A,
            int(self.ptr_B),
            stride_B,
            4 # mma_promotion_interval
        )

        # Set of mainloop arguments needed for this kernel
        mainloop = self.operation.rt_module.mainloop_args.from_generic_mainloop_args(generic_args)

        epilogue = self.operation.rt_module.epilogue_args(
            self.output_op,
            int(self.ptr_C),
            stride_C,
            int(self.ptr_D),
            stride_D,
        )

        # Set hardware info
        hw_info = self.operation.rt_module.hw_info(0, device_sm_count())

        self.arguments = self.operation.argument_type(
            self.gemm_mode,
            problem_size_,
            mainloop,
            epilogue,
            hw_info,
        )
        return self.arguments

    def initialize(self):
        # get the host and evice workspace
        device_workspace_size = self.operation.rt_module.get_device_workspace_size(self)

        if device_workspace_size > 0:
            self.workspace_buffer = device_mem_alloc(device_workspace_size)
            workspace_ptr = self.workspace_buffer.ptr
            err, = cuda.cuMemsetD32(
                workspace_ptr, 0, device_workspace_size // 4)
        else:
            workspace_ptr = None

        device_workspace = 0
        if workspace_ptr is not None and self.gemm_mode == cutlass_bindings.gemm.Mode.GemmSplitKParallel:
            # in GEMM splik-K parallel, the D pointer is redirected
            # to the workspace
            self.ptr_D = cuda.CUdeviceptr(workspace_ptr)
        elif workspace_ptr is not None and self.gemm_mode == cutlass_bindings.gemm.Mode.Gemm:
            # in GEMM split-K serial
            device_workspace = workspace_ptr

        self.get_arguments()
        res_arg = self.operation.rt_module.get_args(
            ctypes.byref(self.arguments),
            ctypes.c_void_p(int(device_workspace)),
        )
        host_workspace = bytearray(res_arg.contents)

        grid = self.operation.rt_module.get_grid_shape(
            ctypes.byref(self.arguments),
            ctypes.c_void_p(int(device_workspace)),
        )
        block = self.operation.rt_module.get_block_shape()

        device_workspace = None

        self.host_workspace = host_workspace
        self.device_workspace = device_workspace
        self.launch_config = LaunchConfiguration(
            [grid.x, grid.y, grid.z],
            [block.x, block.y, block.z],
            self.operation.rt_module.shared_memory_capacity,
        )


def GemmArguments(
    operation: "GemmOperation", problem_size: "cutlass_bindings.gemm.GemmCoord",
    A: "Tensor", B: "Tensor", C: "Tensor", D: "Tensor",
    gemm_mode: "cutlass_bindings.gemm.Mode" = cutlass_bindings.gemm.Mode.Gemm, **kwargs):
    """
    Argument wrapper for GEMM in CUTLASS 2 or 3. It returns either 2x arguments
    or 3x arguments depending on the `arch` field specified in `operation`.

    :param operation: the GEMM operation to take the argument
    :type operation: :class:`cutlass.backend.GemmOperationUniversal` |
     :class:`cutlass.backend.GemmOperationGrouped`

    :param problem_size: GEMM problem size gemm(M, N, K)
    :type operation: :class:`cutlass_bindings.gemm.GemmCoord`

    :param A: tensor A
    :type A: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param B: tensor B
    :type B: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param C: tensor C
    :type C: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param D: tensor D
    :type D: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param gemm_mode: GEMM mode
    :type gemm_mode: :class:`cutlass_bindings.gemm.Mode`

    :param output_op: output operator, optional
    :type output_op: :class:`cutlass.backend.LinearCombinationFunctorArguments`
    """
    if isinstance(operation.swizzling_functor, cutlass_bindings.ThreadblockSwizzleStreamK):
        if operation.api == ApiVersion.v3x:
            raise Exception("Stream K is currently only supported in CUTLASS 2.x")
        ArgClass = GemmArguments2xStreamK
    else:
        ArgClass = GemmArguments3x if operation.api == ApiVersion.v3x else GemmArguments2x
    return ArgClass(operation, problem_size, A, B, C, D, gemm_mode, **kwargs)


class GemmGroupedArguments:
    """
    Argument wrapper for GEMM Grouped. It encodes problem information and
    user-provide tensors into the kernel's argument

    :param operation: the GEMM Grouped operation to take the argument
    :type operation: :class:`cutlass.backend.GemmOperationGrouped`

    :param problem_size: list of GEMM problem size gemm(M, N, K)
    :type operation: list[:class:`cutlass_bindings.gemm.GemmCoord`]

    :param A: list of tensor A
    :type A: list[cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray]

    :param B: list of tensor B
    :type B: list[cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray]

    :param C: list of tensor C
    :type C: list[cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray]

    :param D: list of tensor D
    :type D: list[cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray]

    :param output_op: output operator, optional
    :type output_op: :class:`cutlass.backend.LinearCombinationFunctorArguments`
    """

    def __init__(
        self, operation: "GemmOperationGrouped", problem_sizes: "list[cutlass_bindings.gemm.GemmCoord]",
        A: "list[Tensor]", B: "list[Tensor]", C: "list[torch.Tensor]", D: "list[Tensor]", **kwargs):
        # get number of problems in the group
        self.problem_count = len(problem_sizes)

        # check the input arguments
        assert len(A) == self.problem_count
        assert len(B) == self.problem_count
        assert len(C) == self.problem_count
        assert len(D) == self.problem_count

        problem_size_host = []
        self.ptr_A_host = []
        self.ptr_B_host = []
        self.ptr_C_host = []
        self.ptr_D_host = []

        lda_host = []
        ldb_host = []
        ldc_host = []
        ldd_host = []

        self.partitions = 1

        self.operation = operation

        # get the threadblock
        threadblock_shape = operation.tile_description.threadblock_shape
        self.threadblock_shape = cutlass_bindings.gemm.GemmCoord(
            threadblock_shape[0],
            threadblock_shape[1],
            threadblock_shape[2],
        )
        self.threadblock_swizzle = operation.swizzling_functor

        self.total_tiles = 0

        self.gemm_arguments = []

        # process the input arguments
        for idx, problem_size in enumerate(problem_sizes):
            M, N, K = problem_size.m(), problem_size.n(), problem_size.k()
            temp_argument = GemmArguments2x(
                operation=operation,
                problem_size=cutlass_bindings.gemm.GemmCoord(M, N, K),
                A=A[idx], B=B[idx], C=C[idx], D=D[idx])
            self.gemm_arguments.append(temp_argument)

            problem_size_host.append(
                [temp_argument.problem_size.m(),
                 temp_argument.problem_size.n(),
                 temp_argument.problem_size.k()]
            )

            self.ptr_A_host.append(int(temp_argument.ptr_A))
            lda_host.append(temp_argument.lda)

            self.ptr_B_host.append(int(temp_argument.ptr_B))
            ldb_host.append(temp_argument.ldb)

            self.ptr_C_host.append(int(temp_argument.ptr_C))
            ldc_host.append(temp_argument.ldc)

            self.ptr_D_host.append(int(temp_argument.ptr_D))
            ldd_host.append(temp_argument.ldd)

            # get number of tiles
            grid = self.threadblock_swizzle.get_grid_shape(
                self.threadblock_swizzle.get_tiled_shape(
                    temp_argument.problem_size, self.threadblock_shape,
                    temp_argument.batch_count)
            )
            self.total_tiles += grid.x * grid.y * grid.z

        self.problem_size_buffer = todevice(problem_size_host, np.int32)
        self.ptr_A_buffer = todevice(self.ptr_A_host, np.int64)
        self.ptr_B_buffer = todevice(self.ptr_B_host, np.int64)
        self.ptr_C_buffer = todevice(self.ptr_C_host, np.int64)
        self.ptr_D_buffer = todevice(self.ptr_D_host, np.int64)

        self.lda_buffer = todevice(lda_host, np.int64)
        self.ldb_buffer = todevice(ldb_host, np.int64)
        self.ldc_buffer = todevice(ldc_host, np.int64)
        self.ldd_buffer = todevice(ldd_host, np.int64)

        if "output_op" in kwargs.keys():
            self.alpha = kwargs["output_op"].alpha
            self.beta = kwargs["output_op"].beta
        else:
            self.alpha = 1.0
            self.beta = 0.0

        if "output_op" in kwargs.keys():
            self.output_op = kwargs["output_op"]
        else:
            self.output_op = self.operation.epilogue_type(1.0, 0.0)

        # get host problem size
        self.host_problem_size_ptr = np.array(problem_size_host, dtype=np.int32).__array_interface__["data"][0]

        self.arguments = self.get_arguments()

        self.initialize()

    def get_arguments(self):
        return self.operation.argument_type(
            self.problem_size_buffer.ptr,
            self.problem_count,
            self.total_tiles,
            self.output_op,
            self.ptr_A_buffer.ptr,
            self.ptr_B_buffer.ptr,
            self.ptr_C_buffer.ptr,
            self.ptr_D_buffer.ptr,
            self.lda_buffer.ptr,
            self.ldb_buffer.ptr,
            self.ldc_buffer.ptr,
            self.ldd_buffer.ptr,
            ctypes.c_void_p(int(self.host_problem_size_ptr)),
        )

    def initialize(self):
        # get launch configuration
        launch_config = self.operation.rt_module.plan(self)

        # get the host and evice workspace
        device_workspace_size = self.operation.rt_module.get_device_workspace_size(self)

        if device_workspace_size > 0:
            self.workspace_buffer = device_mem_alloc(device_workspace_size)
            workspace_ptr = self.workspace_buffer.ptr
            err, = cuda.cuMemsetD32(
                workspace_ptr, 0, device_workspace_size // 4)
        else:
            workspace_ptr = None

        if self.operation.precompute_mode == SchedulerMode.Host:
            device_workspace_ptr = self.operation.rt_module.host_precompute(
                self, self.operation.rt_module.get_workspace_size(self),)
        else:
            device_workspace_ptr = 0

        result = self.operation.rt_module.get_args(
            ctypes.byref(self.arguments),
            self.total_tiles,
            ctypes.c_void_p(int(device_workspace_ptr)),
        )
        host_workspace = bytearray(result.contents)

        device_workspace = None

        self.host_workspace = host_workspace
        self.device_workspace = device_workspace
        self.launch_config = launch_config

    def sync(self):
        err, = cudart.cudaDeviceSynchronize()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA Error %s" % str(err))
        for arg in self.gemm_arguments:
            arg.sync(stream_sync=False)


################################################################################
# Base class for GEMM runtime module
################################################################################


class GemmRTbase(ExecutableOperation):
    """
    GemmRT manages the CUTLASS runtime components
    """

    KernelTemplate = r"""
extern "C"
__global__ void
${operation_name}(${operation_name}${operation_suffix}::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  ${operation_name}${operation_suffix}::SharedStorage *shared_storage =
      reinterpret_cast<${operation_name}${operation_suffix}::SharedStorage *>(SharedStorageBase);

  ${operation_name}${operation_suffix}::invoke(params, *shared_storage);
}
  """

    def __init__(self, operation: "GemmOperation"):
        super().__init__(operation)

        self.operation = operation
        threadblock_shape = operation.tile_description.threadblock_shape
        self.threadblock_shape = cutlass_bindings.gemm.GemmCoord(
            threadblock_shape[0], threadblock_shape[1], threadblock_shape[2])
        self.threadblock_swizzle = operation.swizzling_functor

        # Threads per threadblock
        self.threads = operation.tile_description.num_threads

    def emit(self):
        return self.emitter.emit(self.operation)

    def can_implement(self, configuration, arguments):
        raise NotImplementedError()

    def get_host_workspace_size(self, arguments):
        raise NotImplementedError()

    def get_device_workspace_size(self, arguments):
        return 0

    def initialize(self):
        err, = cuda.cuFuncSetAttribute(
            self.kernel,
            attrib=cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value=self.shared_memory_capacity)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                f"CUDA error on call to cuFuncSetAttribute: {cuda.cuGetErrorString(err)[1]}"
            )


################################################################################
# Runtime module for GEMM Universal
################################################################################


class GemmRTUniversal(GemmRTbase):
    """
    GemmRTUniversal manages the CUTLASS runtime components
    """

    HostTemplate = r"""
extern "C" {
  // Get the size of params in bytes
  int ${operation_name}_get_param_size(){
    return sizeof(${operation_name}${operation_suffix}::Params);
  }

  // Get the size of dynamic shared memory in bytes
  int ${operation_name}_shared_memory_size() {
    return int(sizeof(${operation_name}${operation_suffix}::SharedStorage));
  }

  // Get the params as byte array
  char* ${operation_name}_get_params(${operation_name}_base::Arguments* argument, int* workspace){
    ${operation_name}_base::Params* params;
    params = new ${operation_name}_base::Params(*argument,
                                                -1, // SM count. Only used for stream-K
                                                -1  // Occupancy. Only used for stream-K
                                                );

    // Semaphore holds the pointer to the workspace in the Params struct
    params->semaphore = workspace;

    char *bytes = ((char*)(params));
    char *output = new char[sizeof(${operation_name}_base::Params)];
    for (unsigned int i = 0; i < sizeof(${operation_name}_base::Params); i ++)
        output[i] = bytes[i];

    return output;
  }
}
  """

    def __init__(self, operation: "GemmOperation"):
        super(GemmRTUniversal, self).__init__(operation)
        self.emitter = EmitGemmUniversalInstance(
            "_type", operation.direct_store, operation.visitor)

        self.argument_type, self.epilogue_type = get_gemm_arguments(operation.epilogue_functor)
        self.argtype = [
            ctypes.POINTER(self.argument_type),
            ctypes.POINTER(GemmCoord_), ctypes.c_int, ctypes.c_void_p
        ]

    def plan(self, arguments):
        grid = self.threadblock_swizzle.get_tiled_shape(
            arguments.problem_size, self.threadblock_shape, arguments.batch_count
        )

        gemm_k_size = arguments.problem_size.k()
        if arguments.gemm_mode in [cutlass_bindings.gemm.Mode.Gemm, cutlass_bindings.gemm.Mode.GemmSplitKParallel]:
            alignk = max(max(128 // DataTypeSize[self.operation.A.element],
                         128 // DataTypeSize[self.operation.B.element]), 1)

            gemm_k_size = (((arguments.problem_size.k() + arguments.batch_count - 1) //
                           arguments.batch_count + alignk - 1) // alignk) * alignk

            if gemm_k_size:
                grid_z = (arguments.problem_size.k() + gemm_k_size - 1) // gemm_k_size
                grid = cutlass_bindings.gemm.GemmCoord(grid.m(), grid.n(), grid_z)

        arguments.grid_tiled_shape = cutlass_bindings.dim3(grid.m(), grid.n(), grid.k())
        grid = self.threadblock_swizzle.get_grid_shape(grid)
        arguments.gemm_k_size = gemm_k_size
        return LaunchConfiguration(
            [grid.x, grid.y, grid.z],
            [self.threads, 1, 1],
            self.shared_memory_capacity)

    def get_device_workspace_size(self, arguments: GemmArguments):
        workspace_bytes = 0
        if arguments.gemm_mode == cutlass_bindings.gemm.Mode.GemmSplitKParallel:
            workspace_bytes = (DataTypeSize[arguments.operation.C.element]
             * arguments.batched_stride_D * arguments.grid_tiled_shape.z // 8)
        elif (arguments.gemm_mode == cutlass_bindings.gemm.Mode.Gemm and
            arguments.split_k_slices > 1):
            workspace_bytes = 4 * arguments.grid_tiled_shape.x * arguments.grid_tiled_shape.y

        return workspace_bytes


class GemmRTUniversalStreamK(GemmRTUniversal):
    """
    Manages the CUTLASS runtime components for 2.x stream K kernels
    """

    HostTemplate = r"""
extern "C" {
  // Get the size of params in bytes
  int ${operation_name}_get_param_size(){
    return sizeof(${operation_name}${operation_suffix}::Params);
  }

  // Get the size of dynamic shared memory in bytes
  int ${operation_name}_shared_memory_size() {
    return int(sizeof(${operation_name}${operation_suffix}::SharedStorage));
  }

  using GemmType = ${operation_name}_base;

  // Get the params as byte array
  char* ${operation_name}_get_params(GemmType::Arguments* argument, int* workspace,
                                     int sm_count, int occupancy) {
    GemmType::Params* params;
    params = new GemmType::Params(*argument, sm_count, occupancy);

    params->init_workspace(workspace);

    char *bytes = ((char*)(params));
    char *output = new char[sizeof(GemmType::Params)];
    for (unsigned int i = 0; i < sizeof(GemmType::Params); i ++)
        output[i] = bytes[i];

    return output;
  }

  // Get the grid shape
  dim3 ${operation_name}_get_grid_shape(GemmType::Arguments* args, int device_sms, int sm_occupancy) {
    typename GemmType::Params params(*args, device_sms, sm_occupancy);
    return params.get_grid_dims();
  }
}
  """

    def __init__(self, operation: "GemmOperation"):
        super(GemmRTUniversalStreamK, self).__init__(operation)
        self.extra_funcs = {
            "get_grid_shape": GemmCoord_,
        }
        self._occupancy = None
        self.argument_type, self.epilogue_type  = get_gemm_arguments_streamk(operation.epilogue_functor)

    @property
    def occupancy(self):
        if self._occupancy is None:
            err, self._occupancy = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                self.kernel, self.threads, self.shared_memory_capacity,
                cuda.CUoccupancy_flags.CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE)

            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    "CUDA error on call to cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: "
                    f"{cuda.cuGetErrorString(err)[1]}")
        return self._occupancy


################################################################################
# Runtime module for GEMM Universal within CUTLASS 3
################################################################################


class GemmRTUniversal3x(GemmRTUniversal):
    """
    Manages the CUTLASS runtime components for 3.x kernels
    """

    KernelTemplate = r"""

using Operator = ${operation_name}${operation_suffix};
extern "C"
__global__ __launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void ${operation_name}(__grid_constant__ typename Operator::Params const params) {
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];

  // Declare pointer to dynamic shared memory.
  Operator op;
  op(params, smem);
}
  """
    HostTemplate = r"""
extern "C" {
  // Get the size of params in bytes
  int ${operation_name}_get_param_size(){
    return sizeof(${operation_name}${operation_suffix}::Params);
  }

  // Get the size of dynamic shared memory in bytes
  int ${operation_name}_shared_memory_size() {
    return ${operation_name}${operation_suffix}::SharedStorageSize;
  }

  using GemmType = ${operation_name}_base;

  // Get the workspace size
  uint64_t ${operation_name}_get_kernel_workspace_size(GemmType::Arguments* argument) {
    return GemmType::get_workspace_size(*argument);
  }

  // Get the params as byte array
  char* ${operation_name}_get_params(GemmType::Arguments* argument, int* workspace){
    GemmType::Params params = GemmType::to_underlying_arguments(*argument, workspace);

    char *bytes = ((char*)(&params));
    char *output = new char[sizeof(GemmType::Params)];
    for (unsigned int i = 0; i < sizeof(GemmType::Params); i ++)
        output[i] = bytes[i];

    return output;
  }

  // Get the total number of blocks for a persistent kernel
  uint64_t ${operation_name}_get_persistent_tiled_blk_shape_mnl(GemmType::ProblemShape problem) {
    auto problem_shape_MNKL = append<4>(problem, Int<1>{});
    auto [problem_blocks_m, problem_blocks_n, problem_blocks_l] =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::get_tiled_cta_shape_mnl(
            problem_shape_MNKL, GemmType::TileShape{}, GemmType::DispatchPolicy::ClusterShape{});
    return problem_blocks_m * problem_blocks_n * problem_blocks_l;
  }

  // Get the grid shape
  dim3 ${operation_name}_get_grid_shape(GemmType::Arguments* args, int* workspace) {
    auto tmp_params = GemmType::to_underlying_arguments(*args, workspace);
    return GemmType::get_grid_shape(tmp_params);
  }

  // Get the block shape
  dim3 ${operation_name}_get_block_shape() {
    return GemmType::get_block_shape();
  }
}
  """

    def __init__(self, operation: "GemmOperation"):
        super(GemmRTUniversal3x, self).__init__(operation)
        self.extra_funcs = {
            "get_grid_shape": dim3_,
            "get_block_shape": dim3_,
            "get_persistent_tiled_blk_shape_mnl": ctypes.c_uint64,
            "get_kernel_workspace_size": ctypes.c_uint64
        }
        self.emitter = EmitGemmUniversalInstance3x("_type")
        self.mainloop_args = get_mainloop_arguments_3x(
            operation.tile_description.kernel_schedule,
            operation.A.element,
            operation.B.element,
            operation.A.alignment,
            operation.B.alignment
        )
        self.argument_type, self.epilogue_args, self.epilogue_type, self.hw_info = get_gemm_arguments_3x(self.mainloop_args, operation.epilogue_functor)

    def get_device_workspace_size(self, arguments: GemmArguments3x):
        return self.get_kernel_workspace_size(ctypes.byref(arguments.get_arguments()))


class EmitGemmUniversalInstance3x:
    """Responsible for emitting a CUTLASS 3 template definition"""

    def __init__(self, operation_suffix=""):
        self.operation_suffix = operation_suffix
        self.includes = [
            "cutlass/cutlass.h",
            "cute/tensor.hpp",
            "cute/atom/mma_atom.hpp",
            "cutlass/numeric_types.h",
            "cutlass/gemm/collective/collective_builder.hpp",
            "cutlass/gemm/kernel/sm90_tile_scheduler.hpp",
            "cutlass/gemm/kernel/gemm_universal.hpp",
            "cutlass/epilogue/collective/collective_builder.hpp",
            "cutlass/epilogue/collective/default_epilogue.hpp",
            "cutlass/epilogue/thread/linear_combination.h"
        ]
        self.gemm_template_kernel = """
using namespace cute;

using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class},
    cute::Shape<cute::_${threadblock_shape_m}, cute::_${threadblock_shape_n}, cute::_${threadblock_shape_k}>,
    cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule}
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    cute::Shape<cute::_${threadblock_shape_m}, cute::_${threadblock_shape_n}, cute::_${threadblock_shape_k}>,
    cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,
    ${stage_count_type},
    ${kernel_schedule}
  >::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    ${tile_scheduler}
>;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""

        self.gemm_template_device = self.gemm_template_kernel + """

// Define device-level operator
using DeviceKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}${operation_suffix}>;
"""

    def emit(self, operation):
        instance_layout_A, instance_layout_B, instance_layout_C, = \
            (operation.A.layout, operation.B.layout, operation.C.layout)

        # Support built-in epilogue functors or user-defined functions
        epilogue_functor = operation.epilogue_functor.emit()

        if operation.tile_description.stages is None or operation.tile_description.stages == 0:
            stage_count_type = "cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>"
        else:
            stage_count_type = "_" + str(operation.tile_description.stages)

        if operation.emission_type == EmissionType.Kernel:
            gemm_template = self.gemm_template_kernel
        else:
            gemm_template = self.gemm_template_device

        kschedule = KernelScheduleType.ScheduleAuto
        eschedule = EpilogueScheduleType.ScheduleAuto
        tschedule = TileSchedulerType.Default
        if operation.tile_description.kernel_schedule is not None:
            kschedule = operation.tile_description.kernel_schedule
        if operation.tile_description.epilogue_schedule is not None:
            eschedule = operation.tile_description.epilogue_schedule
        if operation.tile_description.tile_scheduler is not None:
            tschedule = operation.tile_description.tile_scheduler

        values = {
            "operation_name": operation.procedural_name(),
            "operation_suffix": self.operation_suffix,
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[instance_layout_A],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[instance_layout_B],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[instance_layout_C],
            "element_d": DataTypeTag[operation.epilogue_functor.element_output],
            "layout_d": LayoutTag[instance_layout_C],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "element_epilogue": DataTypeTag[operation.epilogue_functor.element_epilogue],
            "epilogue_vector_length": str(operation.epilogue_functor.epilogue_vector_length),
            "opcode_class": OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "cluster_m": str(operation.tile_description.cluster_shape[0]),
            "cluster_n": str(operation.tile_description.cluster_shape[1]),
            "cluster_k": str(operation.tile_description.cluster_shape[2]),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "align_c": str(operation.C.alignment),
            "align_d": str(operation.C.alignment),
            "stage_count_type": stage_count_type,
            "kernel_schedule": KernelScheduleTag[kschedule],
            "epilogue_schedule": EpilogueScheduleTag[eschedule],
            "tile_scheduler": TileSchedulerTag[tschedule]
        }

        values["epilogue_functor"] = operation.epilogue_functor.emit()
        return SubstituteTemplate(gemm_template, values)


###################################################################################################
# Runtime module for GEMM Grouped
###################################################################################################


class GemmRTGrouped(GemmRTbase):
    """
    GemmRTGrouped manages the CUTLASS runtime components
    """

    KernelTemplate = r"""
extern "C"
__global__ void
${operation_name}(${operation_name}${operation_suffix}::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  ${operation_name}${operation_suffix}::SharedStorage *shared_storage =
      reinterpret_cast<${operation_name}${operation_suffix}::SharedStorage *>(SharedStorageBase);

  ${operation_name}${operation_suffix} op;

  op(params, *shared_storage);
}
  """

    HostTemplate = r"""
  extern "C" {

    // precompute scheduling information
     char * ${operation_name}_precompute(${operation_name}_base::Arguments const &args, int tile_count, size_t workspace_bytes) {
      char* host_workspace = new char[workspace_bytes];
      ${operation_name}_base::ProblemVisitor::host_precompute(
        args.host_problem_sizes,
        args.problem_count,
        args.threadblock_count,
        (void*)host_workspace
      );
      return host_workspace;
    }

    // Get the size of params in bytes
    int ${operation_name}_get_param_size(){
      return sizeof(${operation_name}${operation_suffix}::Params);
    }

    // Get the size of dynamic shared memory in bytes
    int ${operation_name}_shared_memory_size() {
      return int(sizeof(${operation_name}${operation_suffix}::SharedStorage));
    }

    // Get the params as byte array
    char* ${operation_name}_get_params(${operation_name}_base::Arguments* argument, int tile_count, void* workspace=nullptr){
      ${operation_name}_base::Params* params;
      params = new ${operation_name}_base::Params(*argument, workspace, tile_count);

      char *bytes = ((char*)(params));
      char *output = new char[sizeof(${operation_name}_base::Params)];
      for (unsigned int i = 0; i < sizeof(${operation_name}_base::Params); i ++)
          output[i] = bytes[i];

      return output;
    }
  }
  """

    def __init__(self, operation: "GemmOperation"):
        super(GemmRTGrouped, self).__init__(operation)
        self.extra_funcs = {"precompute": None}

        self.emitter = EmitGemmGroupedInstance("_type")
        self.argument_type, self.epilogue_type = get_gemm_grouped_arguments(operation.epilogue_functor)
        self.argtype = [ctypes.POINTER(self.argument_type), ctypes.c_int, ctypes.c_void_p]

    def host_precompute(self, arguments, workspace_bytes):
        self.precompute.argtype = [
            self.argtype[0], ctypes.c_int, ctypes.c_longlong]
        self.precompute.restype = ctypes.POINTER(ctypes.c_byte * workspace_bytes)

        problem_info = self.precompute(
            ctypes.byref(arguments.arguments),
            arguments.total_tiles,
            workspace_bytes)
        problem_info_array = bytearray(problem_info.contents)

        # copy to device memory
        return rmm.DeviceBuffer.to_device(problem_info_array).ptr

    def plan(self, arguments):
        return LaunchConfiguration(
            [arguments.total_tiles, 1, 1],
            [self.threads, 1, 1],
            self.shared_memory_capacity,
        )

    def get_workspace_size(self, arguments):
        if self.operation.precompute_mode == SchedulerMode.Device:
            return 0
        elif self.operation.precompute_mode == SchedulerMode.Host:
            total_tiles = arguments.total_tiles
            entries_per_block = 1
            return 8 * entries_per_block * total_tiles  # three int32_t


################################################################################
# Runtime module for GEMM and grouped GEMM
################################################################################


class GemmOperationBase:
    """
    CUTLASS GEMM operation
    """

    def __init__(
        self, gemm_kind, arch, tile_description: TileDescription,
        A: TensorDescription, B: TensorDescription, C: TensorDescription,
        epilogue_functor, swizzling_functor=cutlass_bindings.IdentitySwizzle1,
        api=ApiVersion.v2x, emission_type=EmissionType.Kernel, **kwargs):
        self.operation_kind: OperationKind = OperationKind.Gemm
        self.arch: int = arch
        self.tile_description: TileDescription = tile_description
        self.gemm_kind: GemmKind = gemm_kind

        self.api = api
        self.prefix = "3x" if self.api == ApiVersion.v3x else ""
        self.emission_type = emission_type

        # Optionally swap the TensorDescriptions for operands A and B and transpose their
        # layouts. This is needed to mimic the transpose performed by device::GemmUniversal.
        # The code below uses deep copy to avoid overwritting the original TensorDescription
        self.switched = (self.api != ApiVersion.v3x and
                         self.emission_type == EmissionType.Kernel and
                         C.layout == cutlass_bindings.ColumnMajor)

        self.A, self.B, self.C = GemmOperationBase.get_operands(A, B, C, self.switched)

        self.epilogue_functor = epilogue_functor
        self.swizzling_functor = swizzling_functor()

        if "direct_store" in kwargs:
            self.direct_store = kwargs["direct_store"]
        else:
            self.direct_store = False
        if "visitor" in kwargs:
            self.visitor = kwargs["visitor"]
        else:
            self.visitor = False

    @staticmethod
    def get_operands(A: TensorDescription, B: TensorDescription, C: TensorDescription, swap: bool):
        """
        Makes copies of A, B, and C, and possibly transposes their order. If ``swap`` is set,
        A and B are swapped, and the layout of A, B, and C are transposed.

        :param A: description of operand A
        :type A: TensorDescription
        :param B: description of operand B
        :type B: TensorDescription
        :param C: description of operand C
        :type C: TensorDescription

        :return: descriptions of operands A, B, and C
        :rtype: tuple[TileDescription]
        """
        if swap:
            A_out = copy.deepcopy(B)
            B_out = copy.deepcopy(A)
            C_out = copy.deepcopy(C)
            A_out.layout = transpose_layout(A_out.layout)
            B_out.layout = transpose_layout(B_out.layout)
            C_out.layout = transpose_layout(C_out.layout)
        else:
            A_out = copy.deepcopy(A)
            B_out = copy.deepcopy(B)
            C_out = copy.deepcopy(C)
        return A_out, B_out, C_out

    def run(self, arguments: GemmArguments) -> cuda.CUresult:
        """
        Configure and launch the cuda kernel with input arguments
        """
        if self.emission_type == EmissionType.Device:
            raise Exception('Running a kernel via PyCUTLASS is only enabled with emission type "Kernel"')

        err = self.rt_module.run(
            arguments.host_workspace,
            arguments.device_workspace,
            arguments.launch_config,
        )

        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA Error %s" % str(err))

        return err

    def free(self):
        if hasattr(self, "workspace_buffer"):
            del self.workspace_buffer

    def is_complex(self):
        complex_operators = [
            MathOperation.multiply_add_complex,
            MathOperation.multiply_add_complex_gaussian,
            MathOperation.multiply_add_complex_fast_f32,
        ]
        return self.tile_description.math_instruction.math_operation in complex_operators

    def is_planar_complex(self):
        return self.gemm_kind in (GemmKind.PlanarComplex, GemmKind.PlanarComplexArray)

    def accumulator_type(self):
        accum = self.tile_description.math_instruction.element_accumulator

        if self.is_complex():
            return get_complex_from_real(accum)

        return accum

    def short_math_name(self):
        if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
            return "g%s" % ShortDataTypeNames[self.accumulator_type()]
        return ShortDataTypeNames[self.accumulator_type()]

    def core_name(self):
        """The basic operation kind is prefixed with a letter indicating the accumulation type."""

        inst_shape = ""
        inst_operation = ""
        intermediate_type = ""

        math_operations_map = {
            MathOperation.xor_popc: "xor",
        }

        if (self.tile_description.math_instruction.opcode_class == cutlass_bindings.OpClass.TensorOp or
            self.tile_description.math_instruction.opcode_class == cutlass_bindings.OpClass.WmmaTensorOp):
            math_op = self.tile_description.math_instruction.math_operation
            math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ""

            if self.tile_description.math_instruction.instruction_shape is not None:
                inst_shape = "%dx%dx%d" % tuple(
                    self.tile_description.math_instruction.instruction_shape)
            else:
                inst_shape = "Default"
            inst_shape += math_op_string

            if (self.tile_description.math_instruction.element_a != self.A.element and
                self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator):
                intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

        return "%s%s%s%s" % (self.short_math_name(), inst_shape, intermediate_type, GemmKindNames[self.gemm_kind])

    def extended_name(self):
        """Append data types if they differ from compute type."""
        if self.is_complex():
            extended_name = "${core_name}"
        else:
            if (self.C.element != self.tile_description.math_instruction.element_accumulator and
                self.A.element != self.tile_description.math_instruction.element_accumulator):
                extended_name = "${element_c}_${core_name}_${element_a}"
            elif (self.C.element == self.tile_description.math_instruction.element_accumulator and
                self.A.element != self.tile_description.math_instruction.element_accumulator):
                extended_name = "${core_name}_${element_a}"
            else:
                extended_name = "${core_name}"

        extended_name = SubstituteTemplate(extended_name, {
            "element_a": DataTypeNames[self.A.element],
            "element_c": DataTypeNames[self.C.element],
            "core_name": self.core_name(),
        })

        return extended_name

    def extended_name_3x(self):
        """Generates a string representing the MMA atom. Assumes accumulator type is C type."""
        extended_name = "{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}".format(
            element_a=DataTypeNames[self.A.element],
            element_b=DataTypeNames[self.B.element],
            element_acc=DataTypeNames[self.tile_description.math_instruction.element_accumulator],
            element_c=DataTypeNames[self.C.element],
            core_name=self.core_name())
        return extended_name

    def layout_name(self):
        if self.is_complex() or self.is_planar_complex():
            return "%s%s" % (
                ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
                ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)]
            )
        return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

    # Generates a short string representing the ABC layout tags (e.g. ntn or tnn)
    def layout_name_3x(self):
        if self.is_complex() or self.is_planar_complex():
            return "{}{}{}".format(
                ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
                ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)],
                ShortComplexLayoutNames[(self.C.layout, self.C.complex_transform)])
        else:
            return "{}{}{}".format(
                ShortLayoutTypeNames[self.A.layout],
                ShortLayoutTypeNames[self.B.layout],
                ShortLayoutTypeNames[self.C.layout])

    # Generates a short string representing underlying kernel schedule type
    def kernel_schedule_name_3x(self):
        if self.tile_description.kernel_schedule is None:
            return KernelScheduleSuffixes[KernelScheduleType.ScheduleAuto]
        else:
            return KernelScheduleSuffixes[self.tile_description.kernel_schedule]

    # Generates a short string representing underlying epilogue schedule type
    def epilogue_schedule_name_3x(self):
        if self.tile_description.epilogue_schedule is None:
            return EpilogueScheduleSuffixes[EpilogueScheduleType.ScheduleAuto]
        else:
            return EpilogueScheduleSuffixes[self.tile_description.epilogue_schedule]

    def procedural_name(self):
        """The full procedural name indicates architecture, extended name, tile size, and layout."""
        opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
        if self.api == ApiVersion.v3x and self.arch >= 90:
            kernel_name_template = "cutlass{p}_sm{ar}_{op}_{ex}_{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{l}_{s}_align{al}{k}{e}"
            return kernel_name_template.format(
                p=self.prefix,
                ar=self.arch,
                op=opcode_class_name,
                ex=self.extended_name_3x(),
                tbm=self.tile_description.threadblock_shape[0],
                tbn=self.tile_description.threadblock_shape[1],
                tbk=self.tile_description.threadblock_shape[2],
                cm=self.tile_description.cluster_shape[0],
                cn=self.tile_description.cluster_shape[1],
                ck=self.tile_description.cluster_shape[2],
                l=self.tile_description.stages,
                s=self.layout_name_3x(),
                al=str(self.A.alignment),
                k=self.kernel_schedule_name_3x(),
                e=self.epilogue_schedule_name_3x()
            )
        else:
            threadblock = self.tile_description.procedural_name()
            return "cutlass{p}_sm{ar}_{op}_{ex}_{tb}_{l}_align{a}".format(
                p=self.prefix,
                ar=self.arch,
                op=opcode_class_name,
                ex=self.extended_name(),
                tb=threadblock,
                l=self.layout_name(),
                a=str(self.A.alignment)
            )

    def configuration_name(self):
        """The full procedural name indicates architecture, extended name, tile size, and layout."""
        return self.procedural_name()


class GemmOperationUniversal(GemmOperationBase):
    def __init__(self, arch, tile_description: TileDescription, A: TensorDescription, B, C,
        epilogue_functor, swizzling_functor=cutlass_bindings.IdentitySwizzle1, **kwargs):
        api = api_version(arch, tile_description.math_instruction.opcode_class, A.element)
        super(GemmOperationUniversal, self).__init__(GemmKind.Universal, arch, tile_description,
                                                     A, B, C, epilogue_functor, swizzling_functor,
                                                     api=api, **kwargs, )
        if api == ApiVersion.v3x:
            if swizzling_functor == cutlass_bindings.ThreadblockSwizzleStreamK:
                raise Exception("Stream K is currently only supported for CUTLASS 2.x kernels")
            self.rt_module = GemmRTUniversal3x(self)
        else:
            if swizzling_functor == cutlass_bindings.ThreadblockSwizzleStreamK:
                self.rt_module = GemmRTUniversalStreamK(self)
            else:
                self.rt_module = GemmRTUniversal(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

    def device_op(self):
        """
        Returns a new GemmOperationUniversal object that is constructed with emission type
        ``EmissionType.Device``. Since the device-emitted kernel does not require swapping,
        any swappng performed by the kernel-emitted operation is reversed.

        :return: operation ready for device-level code emission
        :rtype: GemmUniversalOperation
        """
        A, B, C = GemmOperationBase.get_operands(self.A, self.B, self.C, self.switched)
        return GemmOperationUniversal(self.arch, self.tile_description, A, B, C,
                                      self.epilogue_functor, type(self.swizzling_functor),
                                      emission_type=EmissionType.Device, direct_store=self.direct_store,
                                      visitor=self.visitor)


class GemmOperationGrouped(GemmOperationBase):
    def __init__(self, arch, tile_description: TileDescription, A: TensorDescription, B, C,
        epilogue_functor, swizzling_functor=cutlass_bindings.IdentitySwizzle1, **kwargs):
        super(GemmOperationGrouped, self).__init__(GemmKind.Grouped, arch, tile_description,
                                                   A, B, C, epilogue_functor, swizzling_functor, **kwargs)
        assert "precompute_mode" in kwargs.keys(), "missing keyword arguement 'precompute_mode'."
        self.precompute_mode = kwargs["precompute_mode"]
        self.rt_module = GemmRTGrouped(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

    def device_op(self):
        """
        Returns a new GemmOperationGrouped object that is constructed with emission type
        ``EmissionType.Device``. Since the device-emitted kernel does not require swapping,
        any swappng performed by the kernel-emitted operation is reversed.

        :return: operation ready for device-level code emission
        :rtype: GemmOperationGrouped
        """
        A, B, C = GemmOperationBase.get_operands(self.A, self.B, self.C, self.switched)
        return GemmOperationGrouped(
            self.arch, self.tile_description, A, B, C, self.epilogue_functor,
            type(self.swizzling_functor), emission_type=EmissionType.Device,
            direct_store=self.direct_store, precompute_mode=self.precompute_mode, )


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################


class EmitGemmUniversalInstance:
    """Responsible for emitting a CUTLASS template definition"""

    def __init__(
        self,
        operation_suffix="",
        direct_store=False,
        visitor=False,
    ):
        self.operation_suffix = operation_suffix
        self.direct_store = direct_store
        self.visitor = visitor
        self.includes = [
            "cutlass/cutlass.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "cutlass/gemm/device/gemm.h",
            "cutlass/gemm/device/gemm_universal_adapter.h",
            "cutlass/gemm/kernel/default_gemm_universal.h",
        ]
        if self.visitor:
            self.includes += [
                "gemm/gemm_universal_with_visitor.h",
                "epilogue/epilogue_visitor_with_layernorm.h",
                "epilogue/epilogue_visitor_generic.h",
            ]
        if self.direct_store:
            self.includes.append(
                "cutlass/epilogue/threadblock/default_epilogue_direct_store.h"
            )
        self.gemm_template_kernel = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} : 
  public ${operation_name}_base { };
"""

        self.gemm_template_device = """
// Gemm operator ${operation_name}
using DeviceKernel =
    typename cutlass::gemm::device::GemmUniversal<
        // Data type and layout of operand A
        ${element_a}, ${layout_a},
        // Data type and layout of operand B
        ${element_b}, ${layout_b},
        // Data type and layout of operand C
        ${element_c}, ${layout_c},
        // Data type of accumulator
        ${element_accumulator},
        // Class of operation
        ${opcode_class},
        // Compute capability of the target kernel
        ${arch},
        // Threadblock tile shape
        cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
        // Warp tile shape
        cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
        // Instruction shape
        cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
        // Epilogue functor
        ${epilogue_functor},
        // Swizzling function
        ${swizzling_functor},
        // Number of pipeline stages
        ${stages},
        // Alignment of operands A and B
        ${align_a}, ${align_b},
        // Type of math operation
        ${math_operation},
        // Complex transform types of operands A and B
        ${transform_a}, ${transform_b}
    >;
"""
        self.gemm_template_direct_store = """
// Gemm operator ${operation_name}
using ${operation_name}_default = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

using ${operation_name}_base = 
  cutlass::gemm::kernel::GemmUniversal<
    ${operation_name}_default::Mma,
    cutlass::epilogue::threadblock::DefaultEpilogueDirectStore<
      ${operation_name}_default::Epilogue
    >::Epilogue,
    ${operation_name}_default::ThreadblockSwizzle
  >;

// Define named type
struct ${operation_name}${operation_suffix} : 
  public ${operation_name}_base { };
"""
        self.gemm_template_visitor = """
// Gemm operator ${operation_name}
using ${operation_name}_default = 
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${elementwise_epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

${epilogue_visitor}

using ${operation_name}_Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    ${operation_name}_EpilogueVisitor,
    typename ${operation_name}_default::Epilogue>::Epilogue;

using ${operation_name}_base =
    cutlass::gemm::kernel::GemmUniversalwithEpilogueVisitor<
        ${operation_name}_default::Mma,
        ${operation_name}_Epilogue,
        ${operation_name}_default::ThreadblockSwizzle
    >;

// Define named type
struct ${operation_name}${operation_suffix} : 
  public ${operation_name}_base { };
"""

    def instance_template(self):
        return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
      cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
    >("${operation_name}"));
${compile_guard_end}
"""

    def emit(self, operation):
        threadblock_shape = operation.tile_description.threadblock_shape
        warp_count = operation.tile_description.warp_count

        warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

        instance_layout_A, instance_layout_B, instance_layout_C = \
            (operation.A.layout, operation.B.layout, operation.C.layout)

        if operation.emission_type == EmissionType.Kernel:
            if self.direct_store:
                gemm_template = self.gemm_template_direct_store
            elif self.visitor:
                gemm_template = self.gemm_template_visitor
            else:
                gemm_template = self.gemm_template_kernel
        else:
            gemm_template = self.gemm_template_device

        values = {
            "operation_name": operation.procedural_name(),
            "operation_suffix": self.operation_suffix,
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[instance_layout_A],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[instance_layout_B],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[instance_layout_C],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(operation.tile_description.math_instruction.instruction_shape[0]),
            "instruction_shape_n": str(operation.tile_description.math_instruction.instruction_shape[1]),
            "instruction_shape_k": str(operation.tile_description.math_instruction.instruction_shape[2]),
            "swizzling_functor": operation.swizzling_functor.tag(),
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": ComplexTransformTag[operation.A.complex_transform],
            "transform_b": ComplexTransformTag[operation.B.complex_transform],
            "math_operation": MathOperationTag[operation.tile_description.math_instruction.math_operation],
        }

        if self.visitor:
            values["epilogue_visitor"] = operation.epilogue_functor.emit(operation)
            values["elementwise_epilogue_functor"] = operation.epilogue_functor.elementwise_functor.emit()
        else:
            values["epilogue_functor"] = operation.epilogue_functor.emit()

        return SubstituteTemplate(gemm_template, values)


class EmitGemmGroupedInstance:
    """Responsible for emitting a CUTLASS template definition"""

    def __init__(self, operation_suffix=""):
        self.operation_suffix = operation_suffix
        self.includes = [
            "cutlass/cutlass.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "cutlass/gemm/kernel/gemm_grouped.h",
            "cutlass/gemm/kernel/default_gemm_grouped.h",
        ]
        self.gemm_template_kernel = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${precompute_mode},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""
        self.gemm_template_device = (
            self.gemm_template_kernel
            + """
using DeviceKernel = cutlass::gemm::device::GemmGrouped<${operation_name}_base>;
"""
        )

    def instance_template(self):
        return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmGrouped<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

    def emit(self, operation):
        threadblock_shape = operation.tile_description.threadblock_shape
        warp_count = operation.tile_description.warp_count

        warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

        instance_layout_A, instance_layout_B, instance_layout_C = \
            (operation.A.layout, operation.B.layout, operation.C.layout)

        # Support built-in epilogue functors or user-defined functions
        epilogue_functor = operation.epilogue_functor.emit()

        values = {
            "operation_name": operation.procedural_name(),
            "operation_suffix": self.operation_suffix,
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[instance_layout_A],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[instance_layout_B],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[instance_layout_C],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(operation.tile_description.math_instruction.instruction_shape[0]),
            "instruction_shape_n": str(operation.tile_description.math_instruction.instruction_shape[1]),
            "instruction_shape_k": str(operation.tile_description.math_instruction.instruction_shape[2]),
            "epilogue_functor": epilogue_functor,
            "swizzling_functor": operation.swizzling_functor.tag(),
            "stages": str(operation.tile_description.stages),
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
            "transform_a": ComplexTransformTag[operation.A.complex_transform],
            "transform_b": ComplexTransformTag[operation.B.complex_transform],
            "precompute_mode": SchedulerModeTag[operation.precompute_mode],
            "math_operation": MathOperationTag[operation.tile_description.math_instruction.math_operation],
        }

        if operation.emission_type == EmissionType.Kernel:
            gemm_template = self.gemm_template_kernel
        else:
            gemm_template = self.gemm_template_device

        return SubstituteTemplate(gemm_template, values)

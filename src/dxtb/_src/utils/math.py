# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility: Math
=============

Typed wrappers for PyTorch's linear algebra functions.
"""

from __future__ import annotations

import torch
from tad_mctc.storch.linalg import eighb
import cupy as cp

from dxtb._src.typing import Any, Tensor
from dxtb import timer

__all__ = ["eigh", "eighb", "qr"]


def eigh(matrix: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
    """
    Typed wrapper for PyTorch's :meth:`~torch.linalg.eigh` function.

    Parameters
    ----------
    matrix : Tensor
        Input matrix,

    Returns
    -------
    tuple[Tensor, Tensor]
        Eigenvalues and eigenvectors of the input matrix.
    """    

    if matrix.is_cuda and matrix.size(-1) < 512:
        # Create a fresh DLPack capsule
        dlpack_tensor = torch.utils.dlpack.to_dlpack(matrix)
        cupy_matrix = cp.fromDlpack(dlpack_tensor)
        timer.start("cupy_eigh")
        w, v = cp.linalg.eigh(cupy_matrix, *args, **kwargs)
        timer.stop("cupy_eigh")   
        return (
            torch.utils.dlpack.from_dlpack(w.toDlpack()),
            torch.utils.dlpack.from_dlpack(v.toDlpack())
        )
    else:
        timer.start("torch_eigh")
        w, v = torch.linalg.eigh(matrix, *args, **kwargs)
        timer.stop("torch_eigh")
        return w, v
        # return torch.linalg.eigh(matrix, *args, **kwargs)


def qr(matrix: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
    """
    Typed wrapper for PyTorch's :meth:`~torch.linalg.qr` function.

    Parameters
    ----------
    matrix : Tensor
        Input matrix.

    Returns
    -------
    tuple[Tensor, Tensor]
        Orthogonal matrix and upper triangular matrix of the input matrix.
    """
    return torch.linalg.qr(matrix, *args, **kwargs)

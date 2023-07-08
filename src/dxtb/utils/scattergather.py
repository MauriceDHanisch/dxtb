"""
Torch's `scatter_reduce` and `gather`
=====================================

Wrappers and convenience functions for `torch.scatter_reduce` and
`torch.gather`.
"""

from __future__ import annotations

import warnings
from functools import wraps

import torch

from ..__version__ import __torch_version__
from .._types import Callable, Gather, ScatterOrGather, Tensor
from .tensors import t2int

__all__ = ["scatter_reduce", "wrap_scatter_reduce", "wrap_gather"]


def twice_remove_negative_index(
    func: Callable[[ScatterOrGather, Tensor, int, int, Tensor], Tensor]
) -> Callable[[ScatterOrGather, Tensor, int, int, Tensor], Tensor]:
    """Wrapper for `gather_twice` function that removes negative indices."""

    @wraps(func)
    def wrapper(
        f: ScatterOrGather,
        x: Tensor,
        dim0: int,
        dim1: int,
        idx: Tensor,
        *args: str,
    ) -> Tensor:
        mask = idx >= 0

        if torch.all(mask):
            return func(f, x, dim0, dim1, idx, *args)

        # gathering in two dimensions requires expanding the mask
        return torch.where(
            mask.unsqueeze(-1) * mask.unsqueeze(-2),
            func(f, x, dim0, dim1, torch.where(mask, idx, 0), *args),
            x.new_tensor(0.0),
        )

    return wrapper


@twice_remove_negative_index
def twice(
    func: ScatterOrGather,
    x: Tensor,
    dim0: int,
    dim1: int,
    idx: Tensor,
    *args: str,
) -> Tensor:
    """
    Spread or gather a tensor along two dimensions

    Parameters
    ----------
    f: Callable
        Function to apply (`torch.gather` or `torch.scatter_reduce`)
    x : Tensor
        Tensor to spread/gather
    index : Tensor
        Index to spread/gather along
    dim0 : int
        Dimension to spread/gather along
    dim1 : int
        Dimension to spread/gather along

    Returns
    -------
    Tensor
        Spread/Gathered tensor
    """

    shape0 = [-1] * x.dim()
    shape0[dim0] = x.shape[dim0]
    y = func(
        x,
        dim1,
        idx.unsqueeze(dim0).expand(*shape0),
        *args,
    )

    shape1 = [-1] * y.dim()
    shape1[dim1] = y.shape[dim1]
    z = func(
        y,
        dim0,
        idx.unsqueeze(dim1).expand(*shape1),
        *args,
    )
    return z


# gather


def gather_remove_negative_index(func: Gather) -> Gather:
    """
    Wrapper for `gather` function that removes negative indices.

    Parameters
    ----------
    func : Gather
        `torch.gather`.

    Returns
    -------
    Gather
        Wrapped `torch.gather` (for use as decorator).
    """

    @wraps(func)
    def wrapper(x: Tensor, dim: int, idx: Tensor, *args: str) -> Tensor:
        mask = idx >= 0
        if torch.all(mask):
            return func(x, dim, idx, *args)

        return torch.where(
            mask,
            func(x, dim, torch.where(mask, idx, 0), *args),
            torch.tensor(0, device=x.device, dtype=x.dtype),
        )

    return wrapper


@gather_remove_negative_index
def gather(x: Tensor, dim: int, idx: Tensor) -> Tensor:
    """
    Wrapper for `torch.gather`.

    Parameters
    ----------
    x : Tensor
        Tensor to gather
    dim : int
        Dimension to gather over
    idx : Tensor
        Index to gather over

    Returns
    -------
    Tensor
        Gathered tensor
    """
    return torch.gather(x, dim, idx)


def wrap_gather(x: Tensor, dim: int | tuple[int, int], idx: Tensor) -> Tensor:
    """
    Wrapper for gather function. Also handles multiple dimensions.

    Parameters
    ----------
    x : Tensor
        Tensor to gather
    dim : int | tuple[int, int]
        Dimension to gather over
    idx : Tensor
        Index to gather over

    Returns
    -------
    Tensor
        Gathered tensor
    """

    if idx.ndim > 1:
        if isinstance(dim, int):
            if x.ndim < idx.ndim:
                x = x.unsqueeze(0).expand(idx.size(0), -1)
        else:
            if x.ndim <= idx.ndim:
                x = x.unsqueeze(0).expand(idx.size(0), -1, -1)

    return (
        gather(x, dim, idx)
        if isinstance(dim, int)
        else twice(torch.gather, x, *dim, idx)
    )


# scatter


def scatter_reduce(
    x: Tensor, dim: int, idx: Tensor, *args: str, fill_value: float | int | None = 0
) -> Tensor:  # pragma: no cover
    """

    .. warning::

        `scatter_reduce` is only introduced in 1.11.1 and the API changes in
        v12.1 in a BC-breaking way. `scatter_reduce` in 1.12.1 and 1.13.0 is
        still in beta and CPU-only.

        Related links:
        - https://pytorch.org/docs/1.12/generated/torch.Tensor.scatter_reduce_.\
          html#torch.Tensor.scatter_reduce_
        - https://pytorch.org/docs/1.11/generated/torch.scatter_reduce.html
        - https://github.com/pytorch/pytorch/releases/tag/v1.12.0
          (section "Sparse")

    Thin wrapper for pytorch's `scatter_reduce` function for handling API 
    changes.

    Parameters
    ----------
    x : Tensor
        Tensor to reduce.
    dim : int
        Dimension to reduce over.
    idx : Tensor
        Index to reduce over.
    fill_value : float | int | None
        Value with which the output is inititally filled (reduction units for
        indices not scattered to). Defaults to `0`.

    Returns
    -------
    Tensor
        Reduced tensor.
    """

    if (1, 11, 0) <= __torch_version__ < (1, 12, 0):  # type: ignore
        actual_device = x.device

        # account for CPU-only implementation
        if "cuda" in str(actual_device):
            x = x.to(torch.device("cpu"))
            idx = idx.to(torch.device("cpu"))

        output = torch.scatter_reduce(x, dim, idx, *args)  # type: ignore
        output = output.to(actual_device)
    elif __torch_version__ >= (1, 12, 0) or __torch_version__ >= (2, 0, 0):  # type: ignore
        out_shape = list(x.shape)
        out_shape[dim] = t2int(idx.max()) + 1

        # filling the output is only necessary if the user wants to preserve
        # the behavior in 1.11, where indices not scattered to are filled with
        # reduction inits (sum: 0, prod: 1)
        if fill_value is None:
            out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        else:
            out = torch.full(out_shape, fill_value, device=x.device, dtype=x.dtype)

        # stop warning about beta and possible API changes in 1.12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = torch.scatter_reduce(out, dim, idx, x, *args)  # type: ignore
    else:
        raise RuntimeError(f"Unsupported PyTorch version ({__torch_version__}) used.")

    return output


def wrap_scatter_reduce(
    x: Tensor,
    dim: int | tuple[int, int],
    idx: Tensor,
    reduce: str,
    extra: bool = False,
) -> Tensor:
    """
    Wrapper for `torch.scatter_reduce` that removes negative indices.

    Parameters
    ----------
    x : Tensor
        Tensor to reduce.
    dim : int | (int, int)
        Dimension to reduce over, defaults to -1.
    idx : Tensor
        Index to reduce over.
    reduce : str
        Reduction method, defaults to "sum".
    extra : bool
        If the tensor to reduce contains a extra dimension of arbitrary size
        that is generally different from the size of the indexing tensor
        (e.g. gradient tensors with extra xyz dimension), the indexing tensor
        has to be modified. This feature is only tested for the aforementioned
        gradient tensors and does only work for one dimension.
        Defaults to `False`.

    Returns
    -------
    Tensor
        Reduced tensor.
    """

    # accounting for the extra dimension is very hacky and anything but general
    if extra is True and isinstance(dim, int):
        shp = [*x.shape[:dim], -1, *x.shape[(dim + 1) :]]

        # here, we assume that two dimension in `idx` mean batched mode
        if idx.ndim < 2:
            if x.ndim == 2:
                idx = idx.unsqueeze(-1).expand(*shp)
            elif x.ndim == 3:
                idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*shp)
        else:
            if x.ndim == 3:
                idx = idx.unsqueeze(-1).expand(*shp)
            elif x.ndim == 4:
                idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*shp)

    idx = torch.where(idx >= 0, idx, 0)
    return (
        scatter_reduce(x, dim, idx, reduce)
        if isinstance(dim, int)
        else twice(scatter_reduce, x, *dim, idx, reduce)
    )

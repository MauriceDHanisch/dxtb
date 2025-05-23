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
ABC: SCF Mixer
==============

This module contains the abstract base class for all mixers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import torch

from dxtb import OutputHandler
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Slicer, Tensor

__all__ = ["Mixer", "DEFAULT_OPTS"]


DEFAULT_OPTS = {
    "maxiter": defaults.MAXITER,
    "f_tol": defaults.F_ATOL,
    "x_tol": defaults.X_ATOL,
    "x_tol_max": defaults.X_ATOL_MAX,
    "damp": defaults.DAMP,
    "damp_init": defaults.DAMP_INIT,
    "damp_generations": defaults.DAMP_GENERATIONS,  # for Anderson
    "damp_diagonal_offset": defaults.DAMP_DIAGONAL_OFFSET,  # for Anderson
    "damp_soft_start": defaults.DAMP_SOFT_START,  # for Anderson
}


class Mixer(ABC):
    """
    Abstract base class for mixer.
    """

    label: str
    """Label for the Mixer."""

    iter_step: int
    """Number of mixing iterations taken."""

    options: dict[str, Any]
    """
    Options for the mixer (damping, tolerances, ...).
    - `damp` (float): Mixing parameter, ∈(0, 1), controls the extent of
      mixing. Larger values result in more aggressive mixing. Defaults to
      0.5 according to [Eyert]_.
    - `damp_init` (float): Mixing parameter to use during the initial
       simple mixing steps. Defaults to ``0.01``.
    """

    _delta: Tensor | None
    """Difference between the current and previous systems."""

    _delta_norm: Tensor | None
    """Norm of the difference between the current and previous systems."""

    _batch_mode: int
    """
    Whether the mixer operates in batch mode.
    Inferring batch mode from within the mixer is unreliable as the mixer can
    converge vector- or matrix-valued quantities. Hence, we must set it from
    outside. In the context of dxtb, inference from the `numbers` variable is
    the best/safest option.
    """

    def __init__(
        self, options: dict[str, Any] | None = None, batch_mode: int = 0
    ) -> None:
        self.label = self.__class__.__name__
        self.options = options if options is not None else DEFAULT_OPTS
        self.iter_step = 0
        self._delta = None
        self._delta_norm = None

        # inferring batch mode from shapes of tensor is unreliable, so we
        # explicitly set this information
        self._batch_mode = batch_mode

    def __init_subclass__(cls):
        def _tolerance_threshold_check(func):
            """
            Wrapper to check the validity of the current tolerance value.

            This wraps `__call__` to ensure that a tolerance threshold check
            gets carried out on the first call, i.e. when `iter_step` is
            zero. If the specified tolerance cannot be achieved then a warning
            will be issued and the tolerance will be downgraded to just below
            the precision limit.
            """

            @wraps(func)
            def wrapper(*args, **kwargs):
                self: Mixer = args[0]
                # pylint: disable=W0212
                if self.iter_step != 0:
                    return func(*args, **kwargs)

                dtype = args[1].dtype
                if torch.finfo(dtype).resolution > self.options["x_tol"]:
                    OutputHandler.warn(
                        f"{cls.__name__}: Tolerance value x_tol="
                        f"{self.options['x_tol']} can't be achieved by "
                        f"a {dtype}. DOWNGRADING!",
                        UserWarning,
                    )
                    self.options["x_tol"] = torch.finfo(dtype).resolution * 50

                if torch.finfo(dtype).resolution > self.options["x_tol_max"]:
                    OutputHandler.warn(
                        f"{cls.__name__}: Tolerance value x_tol_max="
                        f"{self.options['x_tol_max']} can't be achieved by "
                        f"a {dtype}. DOWNGRADING!",
                        UserWarning,
                    )
                    self.options["x_tol_max"] = (
                        torch.finfo(dtype).resolution * 50
                    )

                return func(*args, **kwargs)

            return wrapper

        cls.iter = _tolerance_threshold_check(cls.iter)

    def __str__(self) -> str:  # pragma: no cover
        """Returns representative string."""
        return f"{self.__class__.__name__}({self.iter_step}, {self.options})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)

    @abstractmethod
    def iter(self, x_new: Tensor, x_old: Tensor | None = None) -> Tensor:
        """
        Performs the mixing operation & returns the newly mixed system.

        This should contain only the code required to carry out the mixing
        operation.

        Parameters
        ----------
        x_new : Tensor
            New system.
        x_old : Tensor | None, optional
            Old system. Default to ``None``.

        Returns
        -------
        Tensor
            Newly mixed system(s).

        Note
        ----
        The ``x_old`` argument is normally identical to the ``x_mix``
        value returned from the previous iteration, which is stored by the
        class internally. As such, the ``x_old`` argument can be omitted
        from all but the first step if desired.
        """

    @abstractmethod
    def cull(
        self, conv: Tensor, slicers: Slicer = (...,), mpdim: int = 1
    ) -> None:
        """
        Purge selected systems from the mixer.

        This is useful when a subset of systems have converged during mixing.

        Parameters
        ----------
        conv : Tensor
            Tensor with booleans indicating which systems should be culled
            (True) and which should remain (False).
        slicers : Slicer, optional
            New anticipated size of future inputs excluding the batch
            dimension. This is used to allow superfluous padding values to
            be removed form subsequent inputs. Defaults to `(...,)`.
        mpdim : int, optional
            Number of dimensions for the multipole moments. Defaults to `1`,
            i.e., monopole only. `2` additionally includes dipole contributions.
            `3` includes monopole, dipole and quadrupole contributions.
        """

    @property
    def delta(self) -> Tensor:
        """
        Difference between the current and previous systems.

        This may need to be locally overridden if `_delta` needs to be
        reshaped prior to it being returned.
        """
        if self._delta is None:
            raise RuntimeError("Mixer has no been started yet.")
        return self._delta

    @property
    def delta_norm(self) -> Tensor:
        """
        Norm of the difference between the current and previous systems.

        This may need to be locally overridden if `_delta_norm` needs to be
        reshaped prior to it being returned.
        """
        if self._delta_norm is None:
            raise RuntimeError("Mixer has no been started yet.")
        return self._delta_norm

    @property
    def converged(self) -> Tensor:
        """
        Tensor of bools indicating convergence status of the system(s).

        A system is considered to have converged if the maximum absolute
        difference between the current and previous systems is less than
        the ``tolerance`` value.
        """
        # Check that mixing has been conducted is unnecessary here if we always
        # access delta via the property (and not via ``self._delta``).

        # Compute max norm (L∞) and L2 norm
        if self._batch_mode == 0:
            l2_norm = torch.norm(self.delta, p=None)
            max_norm = torch.norm(self.delta, p=float("inf"))
        else:
            # norm goes over all dims except first (batch dimension)
            dims = tuple(range(-(self.delta.ndim - 1), 0))
            l2_norm = torch.norm(self.delta, dim=dims)
            max_norm = torch.norm(self.delta, p=float("inf"), dim=dims)

        converged_l2 = l2_norm < self.options["x_tol"]
        converged_max = max_norm < self.options["x_tol_max"]

        self._delta_norm = l2_norm

        return converged_max & converged_l2

    def reset(self):
        """
        Resets the mixer to its initial state.

        Calling this function will reset the class & its internal attributes.
        However, any properties set during the initialisation process will be
        retained.
        """
        self.iter_step = 0
        self._x_old = None
        self._delta = None
        self._delta_norm = None

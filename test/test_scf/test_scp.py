"""
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

opts = {"verbosity": 0, "maxiter": 300, "scf_mode": "full_tracking"}


def single(
    dtype: torch.dtype,
    name: str,
    mixer: str,
    tol: float,
    scp_mode: str,
    scf_mode: str,
) -> None:
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["escf"].type(dtype)
    charges = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "LiH", "SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "LYS_xao"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single_medium(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    """Test a few larger system."""
    tol = sqrt(torch.finfo(dtype).eps) * 10
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["S2", "LYS_xao_dist"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single_difficult(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    """These systems do not reproduce tblite energies to high accuracy."""
    tol = 5e-3
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60", "vancoh2"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_single_large(
    dtype: torch.dtype, name: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    single(dtype, name, mixer, tol, scp_mode, scf_mode)


def batched(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    mixer: str,
    scp_mode: str,
    scf_mode: str,
    tol: float,
) -> None:
    dd = {"dtype": dtype}

    sample = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"].type(dtype),
            sample[1]["positions"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample[0]["escf"].type(dtype),
            sample[1]["escf"].type(dtype),
        )
    )
    charges = torch.tensor([0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol, rel=tol) == result.scf.sum(-1)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2", "LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "broyden", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full", "implicit"])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, mixer: str, scp_mode: str, scf_mode: str
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    # full gradient tracking (from TBMaLT) has no Broyden implementation
    if scf_mode == "full" and mixer == "broyden":
        return

    batched(dtype, name1, name2, mixer, scp_mode, scf_mode, tol)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("name3", ["SiH4"])
@pytest.mark.parametrize("mixer", ["anderson", "simple"])
@pytest.mark.parametrize("scp_mode", ["charges", "potential", "fock"])
@pytest.mark.parametrize("scf_mode", ["full"])
def test_batch_three(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    mixer: str,
    scp_mode: str,
    scf_mode: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd = {"dtype": dtype}

    sample = samples[name1], samples[name2], samples[name3]
    numbers = batch.pack(
        (
            sample[0]["numbers"],
            sample[1]["numbers"],
            sample[2]["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample[0]["positions"],
            sample[1]["positions"],
            sample[2]["positions"],
        )
    ).type(dtype)
    ref = batch.pack(
        (
            sample[0]["escf"],
            sample[1]["escf"],
            sample[2]["escf"],
        )
    ).type(dtype)
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.1 if mixer == "simple" else 0.4,
            "mixer": mixer,
            "scf_mode": scf_mode,
            "scp_mode": scp_mode,
            "xitorch_fatol": tol,
            "xitorch_xatol": tol,
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    result = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(ref, abs=tol) == result.scf.sum(-1)
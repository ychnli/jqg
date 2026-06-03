"""
This module specifies how to compute and save diagnostics from the model state.
A diagnostic is specified by a DiagnosticSpec object, which declares how it
should be computed from the model state and auxiliary variables and some useful
metadata.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from jqg.model import AbstractState, Aux, Params

Reduction = Literal["max", "mean", "last", "min"]

DiagnosticCompute = Callable[[Params, AbstractState, Aux], jnp.ndarray]


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str  # unique name for the diagnostic
    common_name: str  # human-readable name for the diagnostic
    dims: tuple[str, ...]  # array dimensions
    units: str  # physical units
    compute: DiagnosticCompute  # function to compute the diagnostic
    interval_reduce: Reduction  # how to reduce the diagnostic over an interval


def cfl(params: Params, state: AbstractState, aux: Aux) -> jnp.ndarray:
    del state
    grid = params.grid
    speed = jnp.maximum(
        jnp.abs(aux.u + params.Ubg[:, None, None]),
        jnp.abs(aux.v),
    )
    return jnp.array(jnp.max(speed) * params.dt / grid.dx)


def ke_spectrum(params: Params, state: AbstractState, aux: Aux) -> jnp.ndarray:
    g = params.grid
    normalization = jnp.array((g.nx * g.ny) ** 2, dtype=jnp.int64)
    return g.kappa_sq * jnp.abs(aux.psi_hat) ** 2 / normalization


def ens_spectrum(params: Params, state: AbstractState, aux: Aux) -> jnp.ndarray:
    g = params.grid
    normalization = jnp.array((g.nx * g.ny) ** 2, dtype=jnp.int64)
    return jnp.abs(state.q_hat) ** 2 / normalization


ALL_DIAGNOSTICS = (
    DiagnosticSpec(
        name="cfl",
        common_name="CFL number",
        dims=("time",),
        units="unitless",
        compute=cfl,
        interval_reduce="max",
    ),
    DiagnosticSpec(
        name="q",
        common_name="PV anomaly",
        dims=("time", "lev", "y", "x"),
        units="s^-1 m^-1",
        compute=lambda params, state, aux: aux.q,
        interval_reduce="mean",
    ),
    DiagnosticSpec(
        name="psi",
        common_name="Streamfunction",
        dims=("time", "lev", "y", "x"),
        units="m^2 s^-1",
        compute=lambda params, state, aux: jnp.fft.irfftn(
            aux.psi_hat, s=(params.grid.ny, params.grid.nx), axes=(-2, -1)
        ),
        interval_reduce="mean",
    ),
    DiagnosticSpec(
        name="u",
        common_name="Zonal velocity",
        dims=("time", "lev", "y", "x"),
        units="m s^-1",
        compute=lambda params, state, aux: aux.u,
        interval_reduce="mean",
    ),
    DiagnosticSpec(
        name="v",
        common_name="Meridional velocity",
        dims=("time", "lev", "y", "x"),
        units="m s^-1",
        compute=lambda params, state, aux: aux.v,
        interval_reduce="mean",
    ),
    DiagnosticSpec(
        name="ke_spec",
        common_name="Kinetic energy spectrum",
        dims=("time", "lev", "l", "k"),
        units="m^2 s^-2",
        compute=ke_spectrum,
        interval_reduce="mean",
    ),
    DiagnosticSpec(
        name="ens_spec",
        common_name="Enstrophy spectrum",
        dims=("time", "lev", "l", "k"),
        units="m^2 s^-2",
        compute=ens_spectrum,
        interval_reduce="mean",
    ),
)


def build_diagnostics(diagnostic_names: Sequence[str]) -> Sequence[DiagnosticSpec]:
    """
    Helper function to build a sequence of DiagnosticSpec objects from a sequence of diagnostic names.
    """
    return tuple(spec for spec in ALL_DIAGNOSTICS if spec.name in diagnostic_names)


DEFAULT_DIAGNOSTICS = build_diagnostics(["q", "psi", "u", "v"])


def compute_diagnostics(
    params: Params,
    state: AbstractState,
    aux: Aux,
    *,
    specs: Sequence[DiagnosticSpec] | None = None,
) -> dict[str, jnp.ndarray]:
    if specs is None:
        specs = ALL_DIAGNOSTICS
    return {spec.name: spec.compute(params, state, aux) for spec in specs}


def init_window_accumulators(
    diag: Mapping[str, jnp.ndarray],
    specs: Sequence[DiagnosticSpec],
) -> dict[str, jnp.ndarray]:
    """Initialize per-window accumulators from the first step's diagnostics."""
    # The first step in each window is folded in here; later steps use
    # :func:`update_window_accumulators`.
    return {spec.name: diag[spec.name] for spec in specs}


def update_window_accumulators(
    acc: Mapping[str, jnp.ndarray],
    diag: Mapping[str, jnp.ndarray],
    specs: Sequence[DiagnosticSpec],
) -> dict[str, jnp.ndarray]:
    """Fold one step's diagnostics into running window accumulators."""
    out: dict[str, jnp.ndarray] = {}
    for spec in specs:
        v = diag[spec.name]
        a = acc[spec.name]
        if spec.interval_reduce == "max":
            out[spec.name] = jnp.maximum(a, v)
        elif spec.interval_reduce == "min":
            out[spec.name] = jnp.minimum(a, v)
        elif spec.interval_reduce == "mean":
            out[spec.name] = a + v
        elif spec.interval_reduce == "last":
            out[spec.name] = v
        else:
            raise ValueError(f"unknown reduction {spec.interval_reduce!r}")
    return out


def finalize_window_accumulators(
    acc: Mapping[str, jnp.ndarray],
    interval_steps: int,
    specs: Sequence[DiagnosticSpec],
) -> dict[str, jnp.ndarray]:
    """Finish one window reduction (e.g. divide mean by window length)."""
    out: dict[str, jnp.ndarray] = {}
    for spec in specs:
        a = acc[spec.name]
        if spec.interval_reduce == "mean":
            out[spec.name] = a / interval_steps
        else:
            out[spec.name] = a
    return out


def aggregate_intervals(
    stacked: Mapping[str, jnp.ndarray],
    interval_steps: int,
    specs: Sequence[DiagnosticSpec],
) -> dict[str, jnp.ndarray]:
    """
    Reduce per-step diagnostic series to one value per interval window.

    Windows are contiguous non-overlapping blocks of ``interval_steps``;
    any trailing substeps fewer than ``interval_steps`` are dropped.
    """
    if interval_steps < 1:
        raise ValueError("interval_steps must be >= 1")

    out: dict[str, jnp.ndarray] = {}
    for spec in specs:
        series = stacked[spec.name]
        nsteps = int(series.shape[0])
        n_windows = nsteps // interval_steps
        if n_windows == 0:
            out[spec.name] = jnp.zeros(
                (0,) + tuple(series.shape[1:]), dtype=series.dtype
            )
            continue
        trim = n_windows * interval_steps
        segment = series[:trim].reshape((n_windows, interval_steps) + series.shape[1:])

        if spec.interval_reduce == "max":
            out[spec.name] = jnp.max(segment, axis=1)
        elif spec.interval_reduce == "min":
            out[spec.name] = jnp.min(segment, axis=1)
        elif spec.interval_reduce == "mean":
            out[spec.name] = jnp.mean(segment, axis=1)
        elif spec.interval_reduce == "last":
            out[spec.name] = segment[:, -1, ...]
        else:
            raise ValueError(f"unknown reduction {spec.interval_reduce!r}")

    return out


def build_diagnostic_coords(
    params: Params,
    *,
    n_windows: int,
    interval_steps: int,
) -> dict[str, np.ndarray]:
    """Physical / spectral coordinates for saved diagnostic fields."""
    grid = params.grid
    dt = float(params.dt)
    time = np.arange(1, n_windows + 1, dtype=np.float64) * interval_steps * dt
    x = np.arange(grid.nx, dtype=np.float64) * grid.dx
    y = np.arange(grid.ny, dtype=np.float64) * grid.dy
    lev = np.array([0, 1], dtype=np.int32)
    k = np.asarray(grid.k[0, :], dtype=np.float64)
    l = np.asarray(grid.l[:, 0], dtype=np.float64)
    return {"time": time, "x": x, "y": y, "lev": lev, "k": k, "l": l}


_COORD_ATTRS = {
    "time": {
        "long_name": "time",
        "units": "s",
        "description": "Seconds since run start, at the end of each output window",
    },
    "x": {"long_name": "zonal distance", "units": "m"},
    "y": {"long_name": "meridional distance", "units": "m"},
    "lev": {
        "long_name": "model layer",
        "units": "1",
        "description": "0=upper layer, 1=lower layer",
    },
    "k": {"long_name": "zonal wavenumber", "units": "rad m^-1"},
    "l": {"long_name": "meridional wavenumber", "units": "rad m^-1"},
}


def diagnostics_to_dataset(
    diagnostics: Mapping[str, jnp.ndarray | np.ndarray],
    specs: Sequence[DiagnosticSpec],
    params: Params,
    *,
    interval_steps: int,
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Build an xarray Dataset from interval-reduced diagnostics."""
    if not specs:
        raise ValueError("specs must be non-empty")

    n_windows = int(np.asarray(diagnostics[specs[0].name]).shape[0])
    coords = build_diagnostic_coords(
        params, n_windows=n_windows, interval_steps=interval_steps
    )

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]] = {}
    for spec in specs:
        data = np.asarray(diagnostics[spec.name])
        expected = tuple(coords[dim].size for dim in spec.dims)
        if data.shape != expected:
            raise ValueError(
                f"diagnostic {spec.name!r}: shape {data.shape} != {expected} "
                f"for dims {spec.dims}"
            )
        data_vars[spec.name] = (
            spec.dims,
            data,
            {"long_name": spec.common_name, "units": spec.units},
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            name: xr.DataArray(
                values,
                dims=(name,),
                attrs=_COORD_ATTRS.get(name, {}),
            )
            for name, values in coords.items()
        },
        attrs=dict(attrs or {}),
    )
    return ds


def write_diagnostics_zarr(
    path: str | Path,
    diagnostics: Mapping[str, jnp.ndarray | np.ndarray],
    specs: Sequence[DiagnosticSpec],
    params: Params,
    *,
    interval_steps: int,
    attrs: Mapping[str, Any] | None = None,
) -> Path:
    """Write interval-reduced diagnostics to a Zarr store via xarray."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = diagnostics_to_dataset(
        diagnostics,
        specs,
        params,
        interval_steps=interval_steps,
        attrs=attrs,
    )
    ds.to_zarr(path, mode="w")
    return path

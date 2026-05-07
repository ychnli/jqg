from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import jax.numpy as jnp

from jqg.model import Aux, Params, State

Reduction = Literal["max", "mean", "last", "min"]

DiagnosticCompute = Callable[[Params, State, Aux], jnp.ndarray]


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    compute: DiagnosticCompute
    interval_reduce: Reduction


def cfl(params: Params, state: State, aux: Aux) -> jnp.ndarray:
    del state
    grid = params.grid
    speed = jnp.maximum(
        jnp.abs(aux.u + params.Ubg[:, None, None]),
        jnp.abs(aux.v),
    )
    return jnp.array(jnp.max(speed) * params.dt / grid.dx)


def pv_spatial_mean(params: Params, state: State, aux: Aux) -> jnp.ndarray:
    """Domain mean of PV anomaly (both layers, single scalar per step)."""
    del params, state
    return jnp.mean(aux.q)


DEFAULT_DIAGNOSTICS: tuple[DiagnosticSpec, ...] = (
    DiagnosticSpec(name="cfl", compute=cfl, interval_reduce="max"),
    DiagnosticSpec(
        name="pv_spatial_mean",
        compute=pv_spatial_mean,
        interval_reduce="mean",
    ),
)


def compute_registered_diagnostics(
    params: Params,
    state: State,
    aux: Aux,
    specs: Sequence[DiagnosticSpec],
) -> dict[str, jnp.ndarray]:
    return {spec.name: spec.compute(params, state, aux) for spec in specs}


def compute_diagnostics(
    params: Params,
    state: State,
    aux: Aux,
    *,
    specs: Sequence[DiagnosticSpec] | None = None,
) -> dict[str, jnp.ndarray]:
    if specs is None:
        specs = DEFAULT_DIAGNOSTICS
    return compute_registered_diagnostics(params, state, aux, specs)


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

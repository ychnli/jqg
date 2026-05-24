from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jqg import QGModel
from jqg.diagnostics import (
    DEFAULT_DIAGNOSTICS,
    aggregate_intervals,
    diagnostics_to_dataset,
    write_diagnostics_zarr,
)
from jqg.solver import run_kernel_interval_jit


def test_nested_scan_matches_aggregate_intervals():
    m = QGModel(nx=16, ny=16, dt=1.0)
    nsteps, interval_steps = 8, 4
    specs = DEFAULT_DIAGNOSTICS

    final_nested, reduced_nested = jax.block_until_ready(
        run_kernel_interval_jit(
            m.params,
            m.state0,
            m.timestepper,
            nsteps,
            interval_steps,
            specs,
        )
    )

    _, stacked = jax.block_until_ready(
        run_kernel_interval_jit(
            m.params,
            m.state0,
            m.timestepper,
            nsteps,
            1,
            specs,
        )
    )
    reduced_ref = aggregate_intervals(stacked, interval_steps, specs)

    for spec in specs:
        assert jnp.allclose(
            reduced_nested[spec.name], reduced_ref[spec.name], rtol=1e-12
        ), spec.name
    assert final_nested.q_hat.shape == (2, 16, 9)


def test_diagnostics_to_dataset():
    m = QGModel(nx=8, ny=8, dt=2.0)
    interval_steps = 4
    n_windows = 2
    diagnostics = {
        "cfl": np.array([0.1, 0.2]),
        "q": np.zeros((n_windows, 2, 8, 8)),
    }
    specs = tuple(s for s in DEFAULT_DIAGNOSTICS if s.name in diagnostics)

    ds = diagnostics_to_dataset(
        diagnostics,
        specs,
        m.params,
        interval_steps=interval_steps,
        attrs={"interval_steps": interval_steps},
    )

    assert ds.sizes["time"] == n_windows
    assert np.allclose(ds["time"].values, [8.0, 16.0])
    assert ds["cfl"].dims == ("time",)
    assert ds["cfl"].attrs["units"] == "unitless"
    assert ds["q"].dims == ("time", "lev", "y", "x")
    assert ds["x"].attrs["units"] == "m"


def test_run_saveto_zarr(tmp_path: Path):
    m = QGModel(nx=8, ny=8, dt=1.0)
    out = tmp_path / "diag.zarr"
    final_state = jax.block_until_ready(m.run(nsteps=8, interval_steps=4, saveto=out))
    assert final_state.q_hat.shape == (2, 8, 5)

    ds = xr.open_zarr(out)
    assert int(ds.attrs["n_windows"]) == 2
    assert int(ds.attrs["interval_steps"]) == 4
    assert ds["cfl"].shape == (2,)
    assert ds["q"].shape == (2, 2, 8, 8)
    assert ds["cfl"].attrs["long_name"] == "CFL number"
    assert ds["time"].attrs["units"] == "s"
    assert "k" in ds.coords
    assert ds["k"].attrs["units"] == "rad m^-1"

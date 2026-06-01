import jax
import numpy as np
from jqg import QGModel
from pathlib import Path

from jqg.utils import plot_single_layer_movie_from_zarr
from jqg.diagnostics import build_diagnostics

from jqg.timesteppers import RK4

name = "barotropic_rossby_wave_rk4"
save_dir = "output/examples"

# enable double precision
jax.config.update("jax_enable_x64", True)


def main():
    out = Path(save_dir) / f"{name}.zarr"

    nx, ny = 256, 256
    Lx, Ly = 1e6, 1e6
    Ld = 15000.0
    hour = 3600  # sec
    day = 24 * hour

    T = 120 * day
    dt = hour * 6
    nsteps = int(T / dt)
    interval_steps = 2

    # initialize PV anomalies to a plane wave
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    xgrid, ygrid = np.meshgrid(x, y, indexing="xy")

    k = 2 * (2 * np.pi / Lx)
    l = 0
    print(f"k * Ld = {k * Ld}")
    print(f"l * Ld = {l * Ld}")

    q_upper = -np.cos(k * xgrid + l * ygrid) * 1e-5
    q_lower = q_upper.copy()

    # initialize a model with no bottom friction, no background zonal flow
    # and equal layer thicknesses
    model = QGModel(
        nx=nx,
        ny=ny,
        dt=dt,
        U1=0,
        U2=0,
        delta=1.0,
        rd=Ld,
        q1=q_upper,
        q2=q_lower,
        r_ekman=0,
        timestepper=RK4(),
    )

    diagnostics = build_diagnostics(["q", "psi", "u", "v"])

    print("Running model...")
    _ = jax.block_until_ready(
        model.run(
            nsteps=nsteps,
            interval_steps=interval_steps,
            saveto=out,
            diagnostics_specs=diagnostics,
        )
    )
    print("done!")

    print("Plotting movie...")
    plot_single_layer_movie_from_zarr(
        out,
        Path(save_dir) / f"{name}.gif",
        title="Upper level PV anomaly",
        vmin=-1e-5,
        vmax=1e-5,
        fps=30,
        dpi=250,
    )
    print("done!")


if __name__ == "__main__":
    main()

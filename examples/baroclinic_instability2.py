import jax
import numpy as np
from jqg import QGModel
from pathlib import Path

from jqg.utils import plot_single_layer_movie_from_zarr
from jqg.diagnostics import build_diagnostics

name = "baroclinic_instability2"
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

    T = 360 * day
    dt = hour # 1 hr timestep
    nsteps = int(T / dt)
    interval_steps = 12 # 12 hr save interval

    # initialize PV anomalies to banded noise
    q_upper = 3e-7 * np.random.rand(ny, nx) + 3e-6 * (np.ones((ny, 1)) * np.random.rand(1, nx))
    q_lower = np.zeros((ny, nx))

    model = QGModel(
        nx=nx,
        ny=ny,
        L=Ly,
        W=Lx,
        dt=dt,
        U1=0.015,
        U2=-0.015,
        delta=1.0,
        rd=Ld,
        q1=q_upper,
        q2=q_lower,
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
        Path(save_dir) / f"{name}.mp4",
        plot_streamfunction=False,
        title="Upper level PV anomaly",
        vmin=-3e-5,
        vmax=3e-5,
        fps=30,
        dpi=250,
    )
    print("done!")


if __name__ == "__main__":
    main()

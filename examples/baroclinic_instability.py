import jax
import numpy as np
from jqg import QGModel
from pathlib import Path

from jqg.utils import plot_single_layer_movie_from_zarr
from jqg.diagnostics import build_diagnostics

name = "baroclinic_instability"
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

    T = 40 * day
    dt = hour / 4  # 15 min timestep
    nsteps = int(T / dt)
    interval_steps = 24  # 6 hourly save interval

    # initialize PV anomalies to a plane wave
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    xgrid, ygrid = np.meshgrid(x, y, indexing="xy")

    k = 10 * (2 * np.pi / Lx)
    l = 0
    print(f"k * Ld = {k * Ld}")
    print(f"l * Ld = {l * Ld}")

    q_upper = -np.cos(k * xgrid + l * ygrid) * 1e-5
    q_lower = np.cos(k * xgrid + l * ygrid) * 1e-5

    model = QGModel(
        nx=nx,
        ny=ny,
        dt=dt,
        U1=0.05,
        U2=-0.05,
        delta=1.0,
        rd=Ld,
        q1=q_upper,
        q2=q_lower,
        r_ekman=0,
        beta=0,
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
        vmin=-2e-5,
        vmax=2e-5,
        fps=30,
        dpi=250,
    )
    print("done!")


if __name__ == "__main__":
    main()

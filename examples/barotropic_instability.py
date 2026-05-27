import jax
import numpy as np
from jqg import QGModel
from pathlib import Path
from jqg.timesteppers import RK4
from jqg.solver import q_hat_tendency
from jqg.utils import plot_single_layer_movie_from_zarr
from jqg.diagnostics import build_diagnostics

name = "barotropic_instability_ab3"
save_dir = "output/examples"

# Force JAX to use GPU (must be called before any JAX operations)
print("Available devices:", jax.devices())
# enable double precision
jax.config.update("jax_enable_x64", True)


def main():
    out = Path(save_dir) / f"{name}.zarr"

    nx, ny = 256, 256
    Lx, Ly = 1e6, 1e6
    Ld = 15000.0
    hour = 3600  # sec
    day = 24 * hour

    T = 180 * day
    dt = hour / 16  # 15 min timestep
    nsteps = int(T / dt)
    interval_steps = 24  # 6 hourly save interval

    # initialize PV anomalies to a jet with white noise
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    _, ygrid = np.meshgrid(x, y, indexing="xy")

    jet_width = Ly / 10

    noise = np.random.normal(0, 1e-7, size=(nx, ny))
    q_upper = np.exp(-((ygrid - Ly / 2) ** 2) / (2 * jet_width**2)) * 1e-5 + noise
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
        # timestepper=RK4(q_hat_tendency),
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
        title="Upper level PV anomaly",
        vmin=-1e-5,
        vmax=1e-5,
        fps=30,
        dpi=250,
    )
    print("done!")


if __name__ == "__main__":
    main()

# Due to the limitations of running files on the server, there have to be two files, even though it is rather bad practice

import jax
import numpy as np
from jqg import QGModel
from pathlib import Path
from jqg.utils import plot_single_layer_movie_from_zarr
from jqg.diagnostics import build_diagnostics
import timeit
from statistics import mean, stdev, correlation

name = "barotropic_instability_ab3"
save_dir = "output/examples"

print("Available devices:", jax.devices())
# enable double precision

# comment or uncomment this to force CPU.
jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)


def benchmark_loop(T=180):

    nx, ny = 256, 256
    Lx, Ly = 1e6, 1e6
    Ld = 15000.0
    hour = 3600  # sec
    day = 24 * hour

    T *= day
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
    )

    diagnostics = build_diagnostics(["q", "psi", "u", "v"])

    start_time = timeit.default_timer()
    _ = jax.block_until_ready(
        model.run(
            nsteps=nsteps,
            interval_steps=interval_steps,
            diagnostics_specs=diagnostics,
        )
    )
    end_time = timeit.default_timer()
    print(f"done in {end_time - start_time} seconds")
    return end_time - start_time


def main():
    for T in [10, 50, 100, 250, 500]:
        times = []
        for _ in range(0, 10):
            times.append(benchmark_loop(T))
        print(f"> {T} : {mean(times) = }")
        print(f"> {T} : {stdev(times) = }")


if __name__ == "__main__":
    main()

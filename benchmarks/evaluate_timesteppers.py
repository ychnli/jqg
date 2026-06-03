"""
Evaluate the speed of AB3 vs RK4 on a baroclinic instability
test case.
"""

import jax
import timeit
import argparse
import json
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np

from jqg.model import QGModel
from jqg.timesteppers import AB3, RK4

jax.config.update("jax_enable_x64", True)

NSTEPS_LIST = [20000, 30000, 40000, 50000]


def eval_timestepper(timestepper, nsteps, verbose=True):
    """
    Run the model for nsteps and do not collect diagnostics
    """

    model = QGModel(nx=64, timestepper=timestepper)

    start_time = timeit.default_timer()
    _ = jax.block_until_ready(
        model.run(nsteps=nsteps, diagnostics_specs=(), saveto=None)
    )
    end_time = timeit.default_timer()

    verbose and print(f"Time taken: {end_time - start_time} seconds")
    return end_time - start_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrials", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def plot_results(results, save_name):
    nsteps_scaling = 1e4
    rk4_results = [
        [int(nsteps) / nsteps_scaling, time]
        for nsteps, data in results.items()
        for time in data["rk4"]
    ]
    ab3_results = [
        [int(nsteps) / nsteps_scaling, time]
        for nsteps, data in results.items()
        for time in data["ab3"]
    ]
    rk4_results = np.array(rk4_results)
    ab3_results = np.array(ab3_results)

    # linear fit
    ab3_fit = scipy.stats.linregress(ab3_results[:, 0], ab3_results[:, 1])
    rk4_fit = scipy.stats.linregress(rk4_results[:, 0], rk4_results[:, 1])
    xvals = np.linspace(ab3_results[0, 0], ab3_results[-1, 0], 100)
    ab3_line = ab3_fit.slope * xvals + ab3_fit.intercept
    rk4_line = rk4_fit.slope * xvals + rk4_fit.intercept

    plt.figure(figsize=(6, 4))
    plt.scatter(ab3_results[:, 0], ab3_results[:, 1], color="tab:blue", label="AB3")
    plt.scatter(rk4_results[:, 0], rk4_results[:, 1], color="tab:orange", label="RK4")

    plt.plot(
        xvals,
        ab3_line,
        color="tab:blue",
        linestyle="--",
        label=f"y = {ab3_fit.slope:.2f}x + {ab3_fit.intercept:.2f}",
    )
    plt.plot(
        xvals,
        rk4_line,
        color="tab:orange",
        linestyle="--",
        label=f"y = {rk4_fit.slope:.2f}x + {rk4_fit.intercept:.2f}",
    )

    plt.legend()
    plt.xlim(1.5, ab3_results[-1, 0] + 0.2)
    plt.ylim(0, np.max(rk4_results) + 2)
    plt.grid()
    plt.xlabel(r"Number of timesteps ($10^{4}$)")
    plt.ylabel("Time (s)")
    plt.title("64 x 64 (CPU)")
    plt.savefig(save_name, dpi=300, bbox_inches="tight")


def main():
    os.makedirs("output/benchmarks", exist_ok=True)
    save_name = "output/benchmarks/timestepper_benchmarks.json"
    args = parse_args()

    if os.path.exists(save_name) and not args.overwrite:
        print(f"Results already exist in {save_name}. Use --overwrite to overwrite.")
        print("Plotting existing results and exiting.")
        results = json.load(open(save_name, "r"))
        plot_results(results, "output/benchmarks/timestepper_benchmarks.png")
        return

    results = {}
    for nsteps in NSTEPS_LIST:
        print(f"Running for {nsteps} steps")
        ab3_times = [
            eval_timestepper(AB3(), nsteps, args.verbose) for _ in range(args.ntrials)
        ]
        rk4_times = [
            eval_timestepper(RK4(), nsteps, args.verbose) for _ in range(args.ntrials)
        ]
        results[nsteps] = {
            "ab3": ab3_times,
            "rk4": rk4_times,
        }

    with open(save_name, "w") as f:
        json.dump(results, f)

    plot_results(results, "output/benchmarks/timestepper_benchmarks.png")

    print("Done!")


if __name__ == "__main__":
    main()

"""
Helper functions for plotting and saving model output.
"""

import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from pathlib import Path
from typing import Callable


def select_upper_layer_pv_anomaly(ds: xr.Dataset):
    return ds["q"].isel(lev=0)


def plot_single_layer_movie_from_zarr(
    save_dir: Path | str,
    output_path: Path | str,
    selector: Callable = select_upper_layer_pv_anomaly,
    plot_streamfunction: bool = True,
    t0: int = 0,
    tf: int | None = None,
    nsteps: int | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    vmin: float = -2e-5,
    vmax: float = 2e-5,
    fps: int = 8,
    dpi: int = 120,
):
    """
    Build an animation from a zarr output, subsetting one field per time step.

    Saves via matplotlib's animation writers:
    - .gif uses Pillow (no ffmpeg needed)
    - .mp4 / .mov need ffmpeg on your PATH

    Args
        save_dir: path to zarr dataset
        output_path: path for the movie file (extension selects the writer)
        selector: function to select the field to plot
        t0: first time index (inclusive)
        tf: last time index (exclusive); default is all times after t0
        nsteps: if set, use tf = t0 + nsteps (overrides default tf)
    """
    ds = xr.open_dataset(save_dir, engine="zarr")

    if tf is None:
        tf = t0 + nsteps if nsteps is not None else ds.time.size

    time_indices = list(range(t0, min(tf, ds.time.size)))
    if not time_indices:
        ds.close()
        raise ValueError("no time steps to plot (check t0, tf, nsteps)")

    fig, ax = plt.subplots(figsize=(6, 5))
    selected_field = selector(ds.isel(time=time_indices[0]))
    lev = selected_field.lev
    mesh = ax.pcolormesh(
        ds["x"].values / 1000,
        ds["y"].values / 1000,
        selected_field,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    streamfunction_plot = None
    if plot_streamfunction:
        streamfunction = ds["psi"].isel(time=time_indices[0]).sel(lev=lev)
        streamfunction_plot = ax.contour(
            ds["x"].values / 1000,
            ds["y"].values / 1000,
            streamfunction,
            colors="black",
            levels=11,
            linewidths=1,
        )

    fig.colorbar(mesh, ax=ax, extend="both")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")

    def _title_for_step(t: int) -> str:
        parts = [p for p in (title, subtitle) if p]
        parts.append(f"t = {int(ds.time.isel(time=t).values / 86400)} days")
        return "\n".join(parts)

    ax.set_title(_title_for_step(time_indices[0]))

    def update(frame: int):
        nonlocal streamfunction_plot
        t = time_indices[frame]
        mesh.set_array(selector(ds.isel(time=t)).values.ravel())
        if streamfunction_plot is not None:
            streamfunction_plot.remove()
            streamfunction = ds["psi"].isel(time=t).sel(lev=lev)
            streamfunction_plot = ax.contour(
                ds["x"].values / 1000,
                ds["y"].values / 1000,
                streamfunction,
                colors="black",
                levels=11,
                linewidths=1,
            )
        ax.set_title(_title_for_step(t))
        return (mesh,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time_indices),
        blit=False,
    )

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix in (".mp4", ".mov", ".mkv"):
        writer = FFMpegWriter(fps=fps)
    else:
        ds.close()
        plt.close(fig)
        raise ValueError(f"unsupported extension {suffix!r}; use .gif, .mp4, or .mov")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    ds.close()
    return output_path

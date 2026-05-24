import jax
import jax.numpy as jnp
from typing import Callable, Sequence

from jqg.diagnostics import (
    DEFAULT_DIAGNOSTICS,
    DiagnosticSpec,
    compute_diagnostics,
    finalize_window_accumulators,
    init_window_accumulators,
    update_window_accumulators,
)
from jqg.model import Aux, Params, State


def _raise_if_cfl_exceeded(cfl) -> None:
    if float(cfl) > 1.0:
        raise ValueError(f"CFL condition violated: {cfl}")


def psi_hat_from_q_hat(params: Params, state: State):
    """
    Compute the spectral streamfunction and the spatial velocities
    from the spectral PV anomaly.

    Args:
        params: Params object
        state: State object
    Returns:
        psi_hat: jnp.ndarray (2, ny, nx//2+1)
        u: jnp.ndarray (2, ny, nx)
        v: jnp.ndarray (2, ny, nx)
    """
    grid = params.grid
    psi_hat = jnp.einsum("ijkl,jkl->ikl", params.M_inv, state.q_hat)

    # compute spectral velocities from spectral streamfunction
    u_hat = -1j * grid.l * psi_hat
    v_hat = 1j * grid.k * psi_hat

    # convert spectral velocities to spatial velocities
    u = jnp.fft.irfftn(u_hat, s=(grid.ny, grid.nx), axes=(-2, -1))
    v = jnp.fft.irfftn(v_hat, s=(grid.ny, grid.nx), axes=(-2, -1))

    return psi_hat, u, v


def q_hat_tendency(params: Params, state: State):
    """
    Compute the tendency of the spectral PV anomaly along with auxiliary quantities
    produced in the process.

    Args:
        params: Params object
        state: State object
    Returns:
        dq_hat_dt: jnp.ndarray (2, ny, nx//2+1) - tendency of spectral PV anomaly
        aux: Aux object
    """

    grid = params.grid

    psi_hat, u, v = psi_hat_from_q_hat(params, state)

    # compute spatial PV anomaly from spectral PV anomaly
    q = jnp.fft.irfftn(state.q_hat, s=(grid.ny, grid.nx), axes=(-2, -1))

    # compute advection of PV anomaly in real space
    uq = (u + params.Ubg[:, None, None]) * q
    vq = v * q

    # convert advection to spectral space
    uq_hat = jnp.fft.rfftn(uq, s=(grid.ny, grid.nx), axes=(-2, -1))
    vq_hat = jnp.fft.rfftn(vq, s=(grid.ny, grid.nx), axes=(-2, -1))

    # compute tendency of spectral PV anomaly due to advection
    dq_hat_dt = -(
        1j * grid.k * uq_hat
        + 1j * grid.l * vq_hat
        + 1j * grid.k * params.dQdy[:, None, None] * psi_hat
    )

    # add bottom frictional drag to tendency
    dq_hat_dt = dq_hat_dt.at[1].add(params.r_ekman * grid.kappa_sq[1] * psi_hat[1])

    aux = Aux(psi_hat=psi_hat, u=u, v=v, q=q)

    return dq_hat_dt, aux


def step(params: Params, state: State, timestepper):
    """
    Advance the model by one timestep.

    Args:
        params: Params object
        state: State object
        timestepper: Timestepper function
    Returns:
        state_new: State object
        diag: Diagnostics dictionary
    """
    # compute tendency of q_hat
    dq_hat_dt, aux = q_hat_tendency(params, state)

    # compute diagnostics
    diag = compute_diagnostics(params, state, aux)

    # update state using chosen timestepper
    state_new = timestepper(dq_hat_dt, state, params)

    return state_new, diag


def run_kernel_interval(
    params: Params,
    state0: State,
    timestepper: Callable,
    nsteps: int,
    interval_steps: int,
    specs: Sequence[DiagnosticSpec],
):
    """
    Advance the model by nsteps, saving diagnostics for each interval window.
    Trailing substeps fewer than ``interval_steps`` are dropped.

    Args:
        params: Model parameters
        state0: Initial state object
        timestepper: Timestepper function
        nsteps: Total timesteps to advance
        interval_steps: Number of timesteps per diagnostic window
        specs: Sequence of DiagnosticSpec objects
    Returns:
        (state, diagnostics): tuple of the final state object and stacked diagnostics
        for each interval window
    """
    if interval_steps < 1:
        raise ValueError("interval_steps must be >= 1")

    n_windows = nsteps // interval_steps

    def outer_step(state: State, _):
        state, diag = step(params, state, timestepper)
        acc = init_window_accumulators(diag, specs)

        def inner_step(carry, _):
            state_in, acc_in = carry
            state_new, diag_step = step(params, state_in, timestepper)
            acc_new = update_window_accumulators(acc_in, diag_step, specs)
            return (state_new, acc_new), None

        if interval_steps > 1:
            (state, acc), _ = jax.lax.scan(
                inner_step,
                (state, acc),
                xs=None,
                length=interval_steps - 1,
            )

        reduced = finalize_window_accumulators(acc, interval_steps, specs)

        if "cfl" in reduced:
            jax.debug.callback(_raise_if_cfl_exceeded, reduced["cfl"])

        return state, reduced

    return jax.lax.scan(outer_step, state0, xs=None, length=n_windows)


run_kernel_interval_jit = jax.jit(
    run_kernel_interval,
    static_argnames=("nsteps", "interval_steps", "timestepper", "specs"),
)

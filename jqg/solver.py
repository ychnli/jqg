import jax
import jax.numpy as jnp
from typing import Callable

from jqg.diagnostics import compute_diagnostics
from jqg.model import Params, State, Aux


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

    aux = Aux(
        psi_hat=psi_hat,
        u=u,
        v=v,
        q=q
    )
    
    return dq_hat_dt, aux



def step(params: Params, state: State, timestepper):
    # compute tendency of q_hat
    dq_hat_dt, aux = q_hat_tendency(params, state)

    # compute diagnostics
    diag = compute_diagnostics(params, state, aux)

    # update state using chosen timestepper
    state_new = timestepper(dq_hat_dt, state, params)
    
    return state_new, diag


def run_kernel(params: Params, state0: State, timestepper: Callable, nsteps: int):
    def scan_step(state, _):
        state_new, diag = step(params, state, timestepper)
        return state_new, diag

    return jax.lax.scan(scan_step, state0, xs=None, length=nsteps)


run_kernel_jit = jax.jit(run_kernel, static_argnames=("nsteps", "timestepper"))
import jax.numpy as jnp
from jqg.solver import Params, State, Aux


def cfl(params: Params, aux: Aux):
    grid = params.grid
    speed = jnp.maximum(
        jnp.abs(aux.u + params.Ubg[:, None, None]),
        jnp.abs(aux.v),
    )
    return jnp.max(speed) * params.dt / grid.dx


def compute_diagnostics(params: Params, state: State, aux: Aux):
    """
    Compute diagnostics from the state.

    Args:
        params: Params object
        state: State object
        aux: Aux object
    Returns:
        diagnostics: dict - diagnostics
    """
    diagnostics = {
        "cfl": cfl(params, aux),
    }
    return diagnostics
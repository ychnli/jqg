"""
This module defines various timesteppers used for the model.
Attached to each timestepper is a dataclass that defines the
state of the model at a given timestep needed to advance
the model to the next timestep. Since all states require the
spectral PV anomaly, each timestepper's state class inherits from
an `AbstractState` class.

Currently we support the following timesteppers:
- AB3: Adams-Bashforth 3rd order timestepper
- RK4: Runge-Kutta 4th order timestepper
"""

import jax.numpy as jnp
from dataclasses import dataclass
from jqg.model import AbstractState, Params
from jqg.solver import q_hat_tendency
from jax.tree_util import register_dataclass


def ab_coefficients(ablevel, dt):
    """
    Helper function to compute the coefficients for the Adams-Bashforth method.

    Args:
        ablevel: index for the order of the Adams-Bashforth method
        dt: timestep

    Returns:
        dt1: coefficient for the first order Adams-Bashforth method
        dt2: coefficient for the second order Adams-Bashforth method
        dt3: coefficient for the third order Adams-Bashforth method
    """
    dt1_fe = dt
    dt2_fe = 0.0
    dt3_fe = 0.0

    dt1_ab2 = 1.5 * dt
    dt2_ab2 = -0.5 * dt
    dt3_ab2 = 0.0

    dt1_ab3 = (23.0 / 12.0) * dt
    dt2_ab3 = (-16.0 / 12.0) * dt
    dt3_ab3 = (5.0 / 12.0) * dt

    dt1 = jnp.where(ablevel == 0, dt1_fe, jnp.where(ablevel == 1, dt1_ab2, dt1_ab3))
    dt2 = jnp.where(ablevel == 0, dt2_fe, jnp.where(ablevel == 1, dt2_ab2, dt2_ab3))
    dt3 = jnp.where(ablevel == 0, dt3_fe, jnp.where(ablevel == 1, dt3_ab2, dt3_ab3))

    return dt1, dt2, dt3


class AB3:
    """
    Adams-Bashforth 3rd order timestepper.

    Since this method requires two previous tendencies, the initial
    state is advanced twice using lower order AB methods.
    """

    @dataclass(frozen=True)
    class AB3State(AbstractState):
        dqdt_p: jnp.ndarray  # tendency from previous step
        dqdt_pp: jnp.ndarray  # tendency from two steps ago
        ablevel: jnp.ndarray  # index for the order of the Adams-Bashforth method

    AB3State = register_dataclass(
        AB3State,
        meta_fields=(),
        data_fields=("q_hat", "dqdt_p", "dqdt_pp", "ablevel"),
    )

    def __init__(self):
        pass

    def create_state(self, q_hat: jnp.ndarray):
        return self.AB3State(
            q_hat=q_hat,
            dqdt_p=jnp.zeros_like(q_hat),
            dqdt_pp=jnp.zeros_like(q_hat),
            ablevel=jnp.array(0),
        )

    def __call__(self, tendency, state: AB3State, params: Params):
        dt1, dt2, dt3 = ab_coefficients(state.ablevel, params.dt)
        q_hat_new = params.grid.spec_filter * (
            state.q_hat + dt1 * tendency + dt2 * state.dqdt_p + dt3 * state.dqdt_pp
        )
        return self.AB3State(
            q_hat=q_hat_new,
            dqdt_p=tendency,
            dqdt_pp=state.dqdt_p,
            ablevel=jnp.minimum(state.ablevel + 1, 2),
        )


class RK4:
    """
    Runge-Kutta 4th order timestepper.
    """

    @dataclass(frozen=True)
    class RK4State(AbstractState):
        pass

    RK4State = register_dataclass(
        RK4State,
        meta_fields=(),
        data_fields=("q_hat",),
    )

    def __init__(self):
        pass

    def create_state(self, q_hat: jnp.ndarray):
        return self.RK4State(q_hat=q_hat)

    def __call__(self, tendency, state: RK4State, params: Params):
        q_hat = state.q_hat
        dt = params.dt
        k1 = tendency
        k2, _ = q_hat_tendency(params, self.create_state(q_hat=q_hat + dt / 2 * k1))
        k3, _ = q_hat_tendency(params, self.create_state(q_hat=q_hat + dt / 2 * k2))
        k4, _ = q_hat_tendency(params, self.create_state(q_hat=q_hat + dt * k3))
        return self.create_state(
            q_hat=params.grid.spec_filter
            * (q_hat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        )

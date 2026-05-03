import jax.numpy as jnp

from jqg.solver import State, Params

def ab_coefficients(ablevel, dt):
    dt1_fe = dt
    dt2_fe = 0.0
    dt3_fe = 0.0

    dt1_ab2 = 1.5 * dt
    dt2_ab2 = -0.5 * dt
    dt3_ab2 = 0.0

    dt1_ab3 = (23.0 / 12.0) * dt
    dt2_ab3 = (-16.0 / 12.0) * dt
    dt3_ab3 = (5.0 / 12.0) * dt

    dt1 = jnp.where(ablevel == 0, dt1_fe,
          jnp.where(ablevel == 1, dt1_ab2, dt1_ab3))
    dt2 = jnp.where(ablevel == 0, dt2_fe,
          jnp.where(ablevel == 1, dt2_ab2, dt2_ab3))
    dt3 = jnp.where(ablevel == 0, dt3_fe,
          jnp.where(ablevel == 1, dt3_ab2, dt3_ab3))

    return dt1, dt2, dt3


def ab3(tendency, state: State, params: Params):
    dt1, dt2, dt3 = ab_coefficients(state.ablevel, params.dt)

    q_hat_new = params.grid.spec_filter * (
        state.q_hat
        + dt1 * tendency
        + dt2 * state.dqdt_p
        + dt3 * state.dqdt_pp
    )

    state_new = State(
        q_hat=q_hat_new,
        dqdt_p=tendency,
        dqdt_pp=state.dqdt_p,
        ablevel=jnp.minimum(state.ablevel + 1, 2),
    )

    return state_new
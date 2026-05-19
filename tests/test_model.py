import jax
import jax.numpy as jnp

from jqg import QGModel


def test_qgmodel_init_shapes():
    nx, ny = 16, 16
    m = QGModel(nx=nx, ny=ny, dt=1.0)
    assert m.grid.nx == nx and m.grid.ny == ny
    assert m.params.dt == 1.0
    nk = nx // 2 + 1
    assert m.state0.q_hat.shape == (2, ny, nk)
    assert jnp.iscomplexobj(m.state0.q_hat)


def test_qgmodel_run_small():
    m = QGModel(nx=16, ny=16, dt=1.0)
    final_state, stacked_diag = jax.block_until_ready(m.run(nsteps=2))
    assert final_state.q_hat.shape == (2, 16, 9)
    assert stacked_diag["cfl"].shape == (2,)
    assert stacked_diag["q"].shape == (2, 2, 16, 16)


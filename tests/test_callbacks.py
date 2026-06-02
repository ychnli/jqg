import jax
from jax.errors import JaxRuntimeError
import pytest

from jqg import QGModel

jax.config.update("jax_enable_x64", True)


def test_raise_if_cfl_exceeded():
    U = 1.0
    nx = 16
    L = 64
    dt = 1.1 * (L / nx) / U 

    m = QGModel(nx=nx, L=L, dt=dt, U1=U, U2=0)

    # this should fail and raise a JaxRuntimeError
    with pytest.raises(JaxRuntimeError, match="CFL condition violated"):
        jax.block_until_ready(m.run(nsteps=10, interval_steps=1))

    
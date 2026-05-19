"""
Tests for the solver module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pyqg

from jqg import QGModel
from jqg.model import State
from jqg.solver import psi_hat_from_q_hat, step

# use double precision for testing
jax.config.update("jax_enable_x64", True)


def test_invert():
    """
    Test inversion of PV anomaly to streamfunction
    """
    nx, ny = 16, 16
    L = 1e6
    W = None
    rd = 15000
    delta = 0.25
    U1 = 0.025
    U2 = 0.0
    beta = 1.5e-11
    r_ekman = 5.787e-7
    q_upper = np.random.randn(ny, nx)
    q_lower = np.random.randn(ny, nx)
    q = np.stack([q_upper, q_lower], axis=0)

    # reference solution
    pyqg_model = pyqg.QGModel(
        nx=nx, ny=ny, L=L, W=W, rd=rd, delta=delta, U1=U1, U2=U2, beta=beta, rek=r_ekman
    )
    pyqg_model.set_q1q2(q_upper, q_lower)

    pyqg_model._invert()

    psi_hat_ref = pyqg_model.ph
    u_ref = pyqg_model.u
    v_ref = pyqg_model.v

    # test solution
    model = QGModel(
        nx=nx,
        ny=ny,
        L=L,
        W=W,
        rd=rd,
        delta=delta,
        U1=U1,
        U2=U2,
        beta=beta,
        r_ekman=r_ekman,
        q1=q_upper,
        q2=q_lower,
    )
    psi_hat, u, v = psi_hat_from_q_hat(model.params, model.state0)

    # cast everything to numpy
    psi_hat_ref = np.array(psi_hat_ref, dtype=np.complex128)
    u_ref = np.array(u_ref, dtype=np.float64)
    v_ref = np.array(v_ref, dtype=np.float64)
    psi_hat = np.array(psi_hat, dtype=np.complex128)
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    assert np.allclose(psi_hat, psi_hat_ref, atol=1e-12)
    assert np.allclose(u, u_ref, atol=1e-12)
    assert np.allclose(v, v_ref, atol=1e-12)


def _jqg_state_from_pyqg(m: pyqg.QGModel) -> State:
    """Match pyqg's PV spectrum and AB3 history before advancing jqg one step."""
    return State(
        q_hat=jnp.asarray(np.array(m.qh), dtype=jnp.complex128),
        dqdt_p=jnp.asarray(np.array(m.dqhdt_p), dtype=jnp.complex128),
        dqdt_pp=jnp.asarray(np.array(m.dqhdt_pp), dtype=jnp.complex128),
        ablevel=jnp.asarray(np.int32(m.ablevel)),
    )


def test_forward_step_resynced_from_pyqg_each_substep():
    """
    Each jqg step starts from pyqg's ``qh`` and the same AB3 carry-over
    buffers (past tendencies).

    That removes trajectory drift from repeated JAX vs FFTW rounding so we
    only see the single-step spectral mismatch (still ~1e-7 on ``q_hat`` for
    ``dt``=7200 s after advection FFTs). Inversion diagnostics then match pyqg's
    ``ph`` once ``_invert()`` has been run (``ph`` is zero right after ``set_q1q2``).
    """
    nx, ny = 16, 16
    L = 1e6
    W = None
    rd = 15000
    delta = 0.25
    U1 = 0.025
    U2 = 0.0
    beta = 1.5e-11
    r_ekman = 5.787e-7
    np.random.seed(42)
    q_upper = np.random.randn(ny, nx) * 1e-6
    q_lower = np.random.randn(ny, nx) * 1e-6

    m = pyqg.QGModel(
        nx=nx,
        ny=ny,
        L=L,
        W=W,
        rd=rd,
        delta=delta,
        U1=U1,
        U2=U2,
        beta=beta,
        rek=r_ekman,
    )
    m.set_q1q2(q_upper, q_lower)
    model = QGModel(
        nx=nx,
        ny=ny,
        L=L,
        W=W,
        rd=rd,
        delta=delta,
        U1=U1,
        U2=U2,
        beta=beta,
        r_ekman=r_ekman,
        q1=q_upper,
        q2=q_lower,
    )

    for i in range(3):
        m._invert()
        psi_hat_ref = np.array(m.ph, dtype=np.complex128)
        state = _jqg_state_from_pyqg(m)
        psi_hat, _, _ = psi_hat_from_q_hat(model.params, state)
        psi_hat = np.array(psi_hat, dtype=np.complex128)
        state, _diag = step(model.params, state, model.timestepper)
        assert np.allclose(psi_hat, psi_hat_ref), (
            f"psi_hat at step {i} after resync, max diff = {np.abs(psi_hat - psi_hat_ref).max()}"
        )
        m._step_forward()
        q_hat = np.array(state.q_hat, dtype=np.complex128)
        q_hat_ref = np.array(m.qh, dtype=np.complex128)
        assert np.allclose(q_hat, q_hat_ref, rtol=1e-5, atol=1e-6), (
            f"q_hat at step {i} after resync, max diff = {np.abs(q_hat - q_hat_ref).max()}"
        )

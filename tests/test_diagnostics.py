import jax.numpy as jnp

from jqg.diagnostics import DiagnosticSpec, aggregate_intervals
from jqg.model import AbstractState, Aux, Params


def _dummy_pv(params: Params, state: AbstractState, aux: Aux):
    del params, state, aux
    return jnp.array(0.0)


def test_aggregate_max_mean_drop_remainder():
    specs = (
        DiagnosticSpec("a", "A", ("time",), "1", _dummy_pv, "max"),
        DiagnosticSpec("b", "B", ("time",), "1", _dummy_pv, "mean"),
    )
    stacked = {
        "a": jnp.array([1.0, 2.0, 3.0, 4.0, 999.0]),
        "b": jnp.array([1.0, 2.0, 3.0, 4.0, 999.0]),
    }
    out = aggregate_intervals(stacked, interval_steps=4, specs=specs)
    assert jnp.allclose(out["a"], jnp.array([4.0]))
    assert jnp.allclose(out["b"], jnp.array([2.5]))


def test_aggregate_last_and_min():
    specs = (
        DiagnosticSpec("a", "A", ("time",), "1", _dummy_pv, "last"),
        DiagnosticSpec("b", "B", ("time",), "1", _dummy_pv, "min"),
    )
    stacked = {
        "a": jnp.array([10.0, 20.0, 30.0, 40.0]),
        "b": jnp.array([3.0, 1.0, 4.0, 2.0]),
    }
    out = aggregate_intervals(stacked, interval_steps=4, specs=specs)
    assert jnp.allclose(out["a"], jnp.array([40.0]))
    assert jnp.allclose(out["b"], jnp.array([1.0]))


def test_aggregate_partial_window_yield_empty():
    specs = (DiagnosticSpec("a", "A", ("time",), "1", _dummy_pv, "max"),)
    stacked = {"a": jnp.array([1.0, 2.0, 3.0])}
    out = aggregate_intervals(stacked, interval_steps=10, specs=specs)
    assert out["a"].shape == (0,)

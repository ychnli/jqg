import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Callable, Sequence

from jax.tree_util import register_dataclass


@dataclass(frozen=True)
class Grid:
    nx: int  # number of grid points in x direction
    ny: int  # number of grid points in y direction
    L: float  # domain size in x direction
    W: float  # domain size in y direction
    dx: float  # grid spacing in x direction
    dy: float  # grid spacing in y direction
    k: jnp.ndarray  # zonal (x) wavenumber (ny, nx//2+1)
    l: jnp.ndarray  # meridional (y) wavenumber (ny, nx//2+1)
    kappa_sq: jnp.ndarray  # squared norm of the wavenumber (ny, nx//2+1)
    spec_filter: jnp.ndarray  # spectral filter function to damp high wavenumbers
    # (ny, nx//2+1)


@dataclass(frozen=True)
class Params:
    grid: Grid  # grid object
    F1: float
    F2: float
    Ubg: jnp.ndarray  # background zonal velocity
    dQdy: jnp.ndarray  # background meridional gradient of PV
    r_ekman: float  # Ekman friction
    dt: float  # timestep
    M_inv: jnp.ndarray  # matrix to convert q to psi


@dataclass(frozen=True)
class AbstractState:
    q_hat: jnp.ndarray


@dataclass(frozen=True)
class Aux:
    psi_hat: jnp.ndarray  # spectral streamfunction
    u: jnp.ndarray  # zonal velocity
    v: jnp.ndarray  # meridional velocity
    q: jnp.ndarray  # spatial PV anomaly


register_dataclass(
    Grid,
    meta_fields=("nx", "ny", "L", "W", "dx", "dy"),
    data_fields=("k", "l", "kappa_sq", "spec_filter"),
)
register_dataclass(
    Params,
    meta_fields=("F1", "F2", "r_ekman", "dt"),
    data_fields=("grid", "Ubg", "dQdy", "M_inv"),
)
register_dataclass(
    Aux,
    meta_fields=(),
    data_fields=("psi_hat", "u", "v", "q"),
)


class QGModel:
    def __init__(
        self,
        nx: int = 64,
        ny: int | None = None,
        L: float = 1e6,
        W: float | None = None,
        beta: float = 1.5e-11,
        rd: float = 15_000.0,
        delta: float = 0.25,
        U1: float = 0.025,
        U2: float = 0.0,
        r_ekman: float = 5.787e-7,
        dt: float = 7200.0,
        filterfac: float = 23.6,
        timestepper: Callable | None = None,
        q1: jnp.ndarray | None = None,
        q2: jnp.ndarray | None = None,
    ):

        self.nx = nx
        self.ny = ny
        self.L = L
        self.W = W
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.U1 = U1
        self.U2 = U2
        self.r_ekman = r_ekman
        self.dt = dt
        self.filterfac = filterfac
        if timestepper is None:
            # default to AB3, which is also used by pyqg
            from jqg.timesteppers import AB3

            timestepper = AB3()

        self.timestepper = timestepper
        self.grid = self._make_grid()
        self.params = self._make_params()
        self.state0 = self._initialize_state(q1=q1, q2=q2)

    def run(
        self,
        nsteps: int,
        *,
        interval_steps: int = 1,
        diagnostics_specs: Sequence | None = None,
        saveto: str | Path | None = None,
    ):
        """Advance the model and collect interval-reduced diagnostics.

        Diagnostics are fused inside a nested scan: an inner loop of
        ``interval_steps`` model steps is reduced per window, and an outer loop
        stacks those windows. Trailing substeps shorter than ``interval_steps``
        are dropped.

        Args:
            nsteps
                Total model timesteps to attempt (truncated to a multiple of
                ``interval_steps``).
            interval_steps
                Number of model timesteps per diagnostic window (default 1).
            diagnostics_specs
                Optional sequence of `DiagnosticSpec` objects.
            saveto
                If set, write diagnostics to this Zarr path and return only the
                final state. Otherwise return ``(final_state, diagnostics)``.

        Returns:
            AbstractState or tuple[AbstractState, dict[str, Array]]
                Final model state, and diagnostics unless ``saveto`` is given.
        """
        import jax

        from jqg.diagnostics import ALL_DIAGNOSTICS, write_diagnostics_zarr
        from jqg.solver import run_kernel_interval_jit

        specs = diagnostics_specs if diagnostics_specs is not None else ALL_DIAGNOSTICS

        # run the model and collect diagnostics
        final_state, diagnostics = jax.block_until_ready(
            run_kernel_interval_jit(
                self.params,
                self.state0,
                self.timestepper,
                nsteps,
                interval_steps,
                specs,
            )
        )

        if saveto is not None:
            n_windows = nsteps // interval_steps
            write_diagnostics_zarr(
                saveto,
                diagnostics,
                specs,
                self.params,
                interval_steps=interval_steps,
                attrs={
                    "nsteps": nsteps,
                    "interval_steps": interval_steps,
                    "n_windows": n_windows,
                    "dt": float(self.dt),
                    "nx": self.grid.nx,
                    "ny": self.grid.ny,
                    "L": float(self.L),
                    "W": float(self.W),
                    "beta": float(self.beta),
                    "rd": float(self.rd),
                    "H1/H2": float(self.delta),
                    "U1": float(self.U1),
                    "U2": float(self.U2),
                    "r_ekman": float(self.r_ekman),
                },
            )
            return final_state

        return final_state, diagnostics

    def _initialize_state(
        self, q1: jnp.ndarray | None = None, q2: jnp.ndarray | None = None
    ) -> AbstractState:
        """
        Initialize the state of the model.

        Args (optional):
            q1: PV anomaly in the upper layer (ny, nx) (default is isotropic Gaussian noise)
            q2: PV anomaly in the lower layer (ny, nx) (default is zero)

        Returns:
            AbstractState: timestepper-specific state object
        """
        # set PV anomaly in real space
        if q1 is None:
            q1 = jnp.asarray(
                1e-7 * np.random.rand(self.ny, self.nx)
                + 1e-6 * (np.ones((self.ny, 1)) * np.random.rand(1, self.nx)),
            )
        if q2 is None:
            q2 = jnp.zeros((self.ny, self.nx))

        # convert to spectral space
        q_hat1 = jnp.fft.rfftn(q1, s=(self.ny, self.nx), axes=(-2, -1))
        q_hat2 = jnp.fft.rfftn(q2, s=(self.ny, self.nx), axes=(-2, -1))

        q_hat = jnp.stack([q_hat1, q_hat2], axis=0)
        state0 = self.timestepper.create_state(q_hat=q_hat)
        return state0

    def _make_grid(self):
        """
        Make a grid for the model.

        Args:
            nx: number of grid points in x direction
            ny: number of grid points in y direction
            L: domain size in x direction (m)
            W: domain size in y direction (m)
            filterfac: filter factor for the spectral filter function
                (default is 23.6, see pyqg documentation for more details)
        Returns:
            Grid: a Grid object
        """

        if self.ny is None:
            self.ny = self.nx
        if self.W is None:
            self.W = self.L

        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # note: since we take the real FFT, the hermitian symmetry implies that
        # f(-k, -l) = f*(k, l). This means that we only need to store values at
        # one half of the domain. By convention, we store the positive wavenumbers
        # for the zonal direction and both positive and negative values for the
        # meridional direction.
        kk = 2 * pi / self.L * jnp.arange(self.nx // 2 + 1)  # zonal wavenumber
        ll = 2 * pi * jnp.fft.fftfreq(self.ny, d=self.dy)  # meridional wavenumber

        k, l_m = jnp.meshgrid(kk, ll)

        kappa_sq = k**2 + l_m**2  # squared norm of the wavenumber

        kstar = jnp.sqrt((k * self.dx) ** 2 + (l_m * self.dy) ** 2)
        cutoff = 0.65 * pi
        spec_filter = jnp.where(
            kstar <= cutoff,
            1.0,
            jnp.exp(-self.filterfac * (kstar - cutoff) ** 4),
        )

        grid = Grid(
            nx=self.nx,
            ny=self.ny,
            L=self.L,
            W=self.W,
            dx=self.dx,
            dy=self.dy,
            k=k,
            l=l_m,
            kappa_sq=kappa_sq,
            spec_filter=spec_filter,
        )
        return grid

    def _make_params(self):
        F1 = self.rd**-2 / (1.0 + self.delta)
        F2 = self.delta * F1

        Ubg = jnp.array([self.U1, self.U2])
        dQdy = jnp.array(
            [
                self.beta + F1 * (self.U1 - self.U2),
                self.beta - F2 * (self.U1 - self.U2),
            ]
        )

        M_inv = self._make_inversion_matrix(F1, F2)

        return Params(
            grid=self.grid,
            F1=F1,
            F2=F2,
            Ubg=Ubg,
            dQdy=dQdy,
            r_ekman=self.r_ekman,
            dt=self.dt,
            M_inv=M_inv,
        )

    def _make_inversion_matrix(self, F1: float, F2: float):
        """
        Make the "inversion matrix" M_inv which is used to compute the
        spectral streamfunction from the spectral PV anomaly.

        For i = 1, 2; j = 1, 2 we have:
            psi_hat_ikl = sum over j of M_inv_ijkl * q_hat_jkl

        Args:
            F1: float
            F2: float
        Returns:
            inversion_matrix: jnp.ndarray (2, 2, ny, nx//2+1)
        """
        kappa_sq = self.grid.kappa_sq
        det = kappa_sq * (kappa_sq + F1 + F2)
        inv_det = jnp.where(det == 0, 0, 1.0 / det)

        A00 = -(kappa_sq + F2) * inv_det
        A01 = -F1 * inv_det
        A10 = -F2 * inv_det
        A11 = -(kappa_sq + F1) * inv_det

        return jnp.stack(
            [
                jnp.stack([A00, A01], axis=0),
                jnp.stack([A10, A11], axis=0),
            ],
            axis=0,
        )

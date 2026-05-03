import jax
import jax.numpy as jnp
from math import pi
from dataclasses import dataclass
from typing import Callable

from jqg.solver import step
from jqg.timesteppers import ab3
from jqg.solver import run_kernel_jit


@dataclass(frozen=True)
class Grid:
    nx: int                 # number of grid points in x direction
    ny: int                 # number of grid points in y direction
    L: float                 # domain size in x direction
    W: float                 # domain size in y direction
    dx: float                # grid spacing in x direction
    dy: float                # grid spacing in y direction
    k: jnp.ndarray          # zonal (x) wavenumber (ny, nx//2+1)
    l: jnp.ndarray          # meridional (y) wavenumber (ny, nx//2+1)
    kappa_sq: jnp.ndarray   # squared norm of the wavenumber (ny, nx//2+1)
    spec_filter: jnp.ndarray # spectral filter function to damp high wavenumbers
                            # (ny, nx//2+1)


@dataclass(frozen=True)
class Params:
    grid: Grid # grid object
    F1: float
    F2: float
    Ubg: jnp.ndarray # background zonal velocity
    dQdy: jnp.ndarray # background meridional gradient of PV 
    r_ekman: float # Ekman friction
    dt: float # timestep
    M_inv: jnp.ndarray # matrix to convert q to psi


@dataclass(frozen=True)
class State:
    q_hat: jnp.ndarray # potential vorticity (PV) anomaly in spectral space 
                       # it has shape (2, ny, nx//2+1), complex
    dqdt_p: jnp.ndarray   # previous tendency
    dqdt_pp: jnp.ndarray  # two-step-old tendency
    ablevel: jnp.ndarray  # 0, 1, or 2


@dataclass(frozen=True)
class Aux:
    psi_hat: jnp.ndarray # spectral streamfunction
    u: jnp.ndarray # zonal velocity
    v: jnp.ndarray # meridional velocity
    q: jnp.ndarray # spatial PV anomaly


class QGModel:
    def __init__(self, nx: int = 64, ny: int | None = None, 
            L: float = 1e6, W: float | None = None, beta: float = 1.5e-11, 
            rd: float = 15_000.0, delta: float = 0.25, U1: float = 0.025, 
            U2: float = 0.0, r_ekman: float = 5.787e-7, dt: float = 7200.0, 
            filterfac: float = 23.6, timestepper: Callable = ab3):

        self.grid = self._make_grid()
        self.params = self._make_params()
        self.state0 = self._initialize_state()
        self.timestepper = timestepper

    def run(self, nsteps: int):
        return run_kernel_jit(self.params, self.state0, self.timestepper, nsteps)

    def _initialize_state(self):
        """
        Initialize the state of the model to isotropic Gaussian noise
        This initialization is identical to the one used in pyqg.
        """
        # set PV anomaly in real space
        q1 = 1e-7 * jnp.random.rand(self.ny, self.nx) + \
            1e-6 * (jnp.ones((self.ny, 1)) * jnp.random.rand(1, self.nx))
        q2 = jnp.zeros((self.ny, self.nx))

        # convert to spectral space
        q_hat1 = jnp.fft.rfftn(q1, s=(self.ny, self.nx), axes=(-2, -1))
        q_hat2 = jnp.fft.rfftn(q2, s=(self.ny, self.nx), axes=(-2, -1))

        state0 = State(
            q_hat = jnp.stack([q_hat1, q_hat2], axis=0),
            dqdt_p = jnp.zeros_like(q_hat1),
            dqdt_pp = jnp.zeros_like(q_hat1),
            ablevel = jnp.array(0),
        )
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
        kk = 2 * pi / self.L * jnp.arange(self.nx // 2 + 1) # zonal wavenumber
        ll = 2 * pi * jnp.fft.fftfreq(self.ny, d=self.dy) # meridional wavenumber

        k, l = jnp.meshgrid(kk, ll)

        kappa_sq = k**2 + l**2 # squared norm of the wavenumber

        kstar = jnp.sqrt((k * self.dx)**2 + (l * self.dy)**2)
        cutoff = 0.65 * pi
        spec_filter = jnp.where(
            kstar <= cutoff,
            1.0,
            jnp.exp(-self.filterfac * (kstar - cutoff)**4),
        )

        grid = Grid(
            nx=self.nx, ny=self.ny, L=self.L, W=self.W, dx=self.dx, dy=self.dy,
            k=k, l=l, kappa_sq=kappa_sq, spec_filter=spec_filter
        )
        return grid

    def _make_params(self):
        F1 = self.rd**-2 / (1.0 + self.delta)
        F2 = self.delta * F1

        Ubg = jnp.array([self.U1, self.U2])
        dQdy = jnp.array([
            self.beta + F1 * (self.U1 - self.U2),
            self.beta - F2 * (self.U1 - self.U2),
        ])

        M_inv = self._make_inversion_matrix()

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

    def _make_inversion_matrix(self):
        """
        Make the "inversion matrix" M_inv which is used to compute the 
        spectral streamfunction from the spectral PV anomaly. 
        
        For i = 1, 2; j = 1, 2 we have:
            psi_hat_ikl = sum over j of M_inv_ijkl * q_hat_jkl

        Args:
            grid: Grid object
            F1: float
            F2: float
        Returns:
            inversion_matrix: jnp.ndarray (2, 2, ny, nx//2+1)
        """
        kappa_sq = self.grid.kappa_sq
        det = kappa_sq * (kappa_sq + self.F1 + self.F2)
        inv_det = jnp.where(det == 0, 0, 1.0 / det)

        A00 = -(kappa_sq + self.F2) * inv_det
        A01 = -self.F1 * inv_det
        A10 = -self.F2 * inv_det
        A11 = -(kappa_sq + self.F1) * inv_det

        return jnp.stack([
            jnp.stack([A00, A01], axis=0),
            jnp.stack([A10, A11], axis=0),
        ], axis=0)



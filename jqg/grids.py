import jax.numpy as jnp
from math import pi
from dataclasses import dataclass

@dataclass(frozen=True)
class Grid:
    nx: int # number of grid points in x direction
    ny: int # number of grid points in y direction
    L: float # domain size in x direction
    W: float # domain size in y direction
    dx: float # grid spacing in x direction
    dy: float # grid spacing in y direction
    k: jnp.ndarray # zonal wavenumber 
    l: jnp.ndarray # meridional wavenumber
    wv2: jnp.ndarray # squared norm of the wavenumber
    filter_: jnp.ndarray # filter function to damp high wavenumbers

def make_grid(nx=64, ny=None, L=1e6, W=None, filterfac=23.6):
    """
    Make a grid for the model.
    
    Args:
        nx: number of grid points in x direction
        ny: number of grid points in y direction
        L: domain size in x direction
        W: domain size in y direction
        filterfac: filter factor for the filter function 
            (default is 23.6, see pyqg documentation for more details)
    Returns:
        Grid: a Grid object
    """

    if ny is None:
        ny = nx
    if W is None:
        W = L

    dx = L / nx
    dy = W / ny

    # note: since we take the real FFT, the hermitian symmetry implies that 
    # f(-k, -l) = f*(k, l). This means that we only need to store values at
    # one half of the domain. By convention, we store the positive wavenumbers
    # for the zonal direction and both positive and negative values for the
    # meridional direction.
    kk = 2 * pi / L * jnp.arange(nx // 2 + 1) # zonal wavenumber
    ll = 2 * pi * jnp.fft.fftfreq(ny, d=dy) # meridional wavenumber

    k, l = jnp.meshgrid(kk, ll)

    wv2 = k**2 + l**2

    kstar = jnp.sqrt((k * dx)**2 + (l * dy)**2)
    cutoff = 0.65 * pi
    filter_ = jnp.where(
        kstar <= cutoff,
        1.0,
        jnp.exp(-filterfac * (kstar - cutoff)**4),
    )

    return Grid(
        nx=nx, ny=ny, L=L, W=W, dx=dx, dy=dy,
        k=k, l=l, wv2=wv2, filter_=filter_
    )
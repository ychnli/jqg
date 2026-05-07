# jqg

Two-layer quasi-geostrophic model in the Jax framework. This follows the same numerics, namely the pseudospectral solver and timestepping, as pyqg.

## Installation (uv)

For machines without GPU:
  ```bash
  uv sync
  ```

For machines with GPUs and the CUDA12 driver.
  ```bash
  uv sync --extra cuda12
  ```
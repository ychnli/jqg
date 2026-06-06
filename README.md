# jqg

Two-layer quasi-geostrophic model on a doubly period rectangular domain implemented in the JAX framework. This largely follows the same numerics, namely the pseudospectral solver and timestepping, as pyqg.

## Installation (uv)

For machines without GPU:
  ```bash
  uv sync
  ```

For machines with GPUs and the CUDA12 driver.
  ```bash
  uv sync --extra cuda12
  ```

## Repository structure

```
jqg/
├── jqg/                         
│   ├── __init__.py
│   ├── model.py                 # user-interfacing model class
│   ├── solver.py                # numerical solver
│   ├── timesteppers.py          # timestepping schemes
│   ├── diagnostics.py           # protocols for saving diagnostics from runs
│   └── utils.py                 # plotting helper functions
├── examples/                    # physical test cases
├── tests/                       # unit test cases 
├── benchmarks/                  # simple benchmarking
├── notebooks/                   # notebooks (scratch space)
├── pyproject.toml               # environment configuration
└── uv.lock                      # environment configuration
```

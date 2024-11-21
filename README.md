# Scientific Computing Tools for Finite Element Methods

This package contains a collection of tools for scientific computing with a focus on finite element methods. The tools are written in Python and are intended to be used in conjunction with the [dolfinx](https://github.com/FEniCS/dolfinx).

Many users that are transitioning from legacy FEniCS to FEniCSx may find the transition difficult due to the lack of some functionalities in FEniCSx.
This package aims to provide some of the functionalities that are missing in FEniCSx.
The package is still in its early stages and many functionalities are still missing.

## Features

- Real-space implementation for usage in DOLFINx (>=v0.8.0)
- Save quadrature functions as point clouds
- Save any function that can tabulate dof coordinates as point clouds.
- Point sources for usage in DOLFINx (>=v0.8.0)
  - Point sources in vector spaces are only supported on v0.9.0, post [DOLFINx PR 3429](https://github.com/FEniCS/dolfinx/pull/3429).
    For older versions, apply one point source in each sub space.
- Simplified wrapper to create MeshTags based on a list of tags and corresponding locator functions.
- Maps between degrees of freedom and vertices: `vertex_to_dofmap` and `dof_to_vertex`
- Blocked Newton Solver
- Function evaluation at specified points

## Installation

The package is partly written in C++ and relies on `dolfinx`. User are encouraged to install `scifem` with `pip` in an environment where `dolfinx` is already installed or with `conda`.

### `pip`
To install the package with `pip` run

```bash
python3 -m pip install scifem --no-build-isolation
```

To install the development version you can run

```bash
python3 -m pip install --no-build-isolation git+https://github.com/scientificcomputing/scifem.git
```

Note that you should pass the flag `--no-build-isolation` to `pip` to avoid issues with the build environment, such as incompatible versions of `nanobind`.

### `conda`

To install the package with `conda` run

```bash
conda install -c conda-forge scifem
```





## Having issues or want to contribute?

If you are having issues, feature request or would like to contribute, please let us know. You can do so by opening an issue on the [issue tracker](https://github.com/scientificcomputing/scifem/issues).

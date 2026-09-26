# Scientific Computing Tools for Finite Element Methods

This package contains a collection of tools for scientific computing with a focus on finite element methods. The tools are written in Python and are intended to be used in conjunction with the [dolfinx](https://github.com/FEniCS/dolfinx).

Many users that are transitioning from legacy FEniCS to FEniCSx may find the transition difficult due to the lack of some functionalities in FEniCSx.
This package aims to provide some of the functionalities that are missing in FEniCSx.

See the [documentation](https://scientificcomputing.github.io/scifem/) for the API reference and runnable examples of the features below.

## Backwards compatibility
We aim to support the last two stable releases of DOLFINx. Sometimes features are supported for longer if need be.

Some features might depend on functionality added to DOLFINx, which means that they won't be backwards compatible.

## Features

### Meshes, tags and submeshes

- Create `MeshTags` from a list of tags and corresponding locator functions.
- Extract a submesh of entities of any co-dimension.
- Locate the interface between two subdomains, and the exterior facets of a subdomain.

### Periodic meshes

- Build a periodic mesh (MPI supported), by merging the vertices on opposite sides of the domain. The periodicity lives in the topology, so a continuous function space on the result is periodic with no constraint matrix, as opposed to [DOLFINx_MPC](https://github.com/jorgensd/dolfinx_mpc).
- Take the vertex pairs from the `$Periodic` section of a gmsh model rather than from the coordinates.
- Move mesh tags onto a periodic mesh, and move a solution back onto the mesh it was built from. Required for post-processing with Pyvista or Paraview.
- Diagnose the two ways an input mesh can be unsuitable, a missing ghost layer and cells that collapse onto each other across a seam.

### Function spaces and degrees of freedom

- Spaces of functions that are constant on each subdomain.
- Maps between degrees of freedom and vertices, in both directions.

### Assembly, sources and boundary conditions

- Assemble a scalar or a norm over all processes in a single call.
- Point sources for usage in DOLFINx (>=v0.8.0).
  - Point sources in vector spaces are only supported on v0.9.0, post [DOLFINx PR 3429](https://github.com/FEniCS/dolfinx/pull/3429).
    For older versions, apply one point source in each sub space.
- Interpolate an expression onto the degrees of freedom of a set of facets.
- Helpers for the PETSc vector operations that surround a solve: zeroing, ghost updates, lifting and boundary conditions.

### Evaluation, interpolation and geometry

- Evaluate a function at arbitrary points, in parallel.
- Find the extrema of a UFL expression within each cell, or over a whole domain.
- Project points onto the closest point of a mesh.
- Build interpolation matrices from any `ufl.core.expr.Expr` into a compatible space.
- Interpolate from a mesh onto a submesh of its facets, and extend a function on such a submesh back into the mesh, zero away from the facets, into Lagrange, H(div) or H(curl) spaces.

### Output

- Save quadrature functions as point clouds.
- Save any function that can tabulate dof coordinates as point clouds.

### Biomedical

- Read MRI data onto a mesh, as a function or as cell tags. Requires the `biomed` extra.

## Installation

The package is partly written in C++ and relies on `dolfinx`. Users are encouraged to install `scifem` with `pip` in an environment where `dolfinx` is already installed or with `conda`.

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

### `spack`
The spack package manager is the recommended way to install scifem, and especially on HPC systems.
For information about the package see: [spack-package: py-scifem](https://packages.spack.io/package.html?name=py-scifem)
First, clone the spack repository and enable spack

```bash
git clone --depth=2 https://github.com/spack/spack.git
# For bash/zsh/sh
. spack/share/spack/setup-env.sh

# For tcsh/csh
source spack/share/spack/setup-env.csh

# For fish
. spack/share/spack/setup-env.fish
```

Next create an environment:

```bash
spack env create scifem_env
spack env activate scifem_env
```
Find the compilers on the system
```bash
spack compiler find
```

and install the relevant packages
```bash
spack add py-scifem+petsc+hdf5+biomed+adios2 ^mpich ^petsc+mumps+hypre ^py-fenics-dolfinx+petsc4py
spack concretize
spack install
```
Finally, note that spack needs some packages already installed on your system. On a clean ubuntu container for example one needs to install the following packages before running spack
```bash
apt update && apt install gcc unzip git python3-dev g++ gfortran xz-utils -y
```
### `conda`

To install the package with `conda` run

```bash
conda install -c conda-forge scifem
```

## Having issues or want to contribute?

If you are having issues, feature request or would like to contribute, please let us know. You can do so by opening an issue on the [issue tracker](https://github.com/scientificcomputing/scifem/issues).

# Scientific Computing Tools for Finite Element Methods

This package contains a collection of tools for scientific computing with a focus on finite element methods. The tools are written in Python and are intended to be used in conjunction with the [dolfinx](https://github.com/FEniCS/dolfinx).

Many users that are transitioning from legacy FEniCS to FEniCSx may find the transition difficult due to the lack of some functionalities in FEniCSx. This package aims to provide some of the functionalities that are missing in FEniCSx. The package is still in its early stages and many functionalities are still missing. 


## Installation

The package is partly written in C++ and relies on `dolfinx`. User are encouraged to install `scifem` in an environment where `dolfinx` is already installed. The package can be installed using `pip`:

```bash
python3 -m pip install scifem --no-build-isolation
```


## Having issues or want to contribute?

If you are having issues, feature request or would like to contribute, please let us know. You can do so by opening an issue on the [issue tracker](https://github.com/scientificcomputing/scifem/issues).



## LICENSE
MIT
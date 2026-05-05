"""Layer for small backward compatibility wrappers for DOLFINx"""

import dolfinx


def get_cmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.CoordinateElement:
    """Get the basix Cmap for the mesh."""
    if callable(mesh.geometry.cmap):
        return mesh.geometry.cmap()
    else:
        return mesh.geometry.cmap

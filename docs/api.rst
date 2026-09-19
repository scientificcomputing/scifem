#############
API reference
#############

The names below are grouped by what they are for. Everything down to `Solvers`_ is
re-exported at the top level, so ``scifem.assemble_scalar`` and
``scifem.assembly.assemble_scalar`` are the same function and the short path is the
documented one. The subpackages are documented under their own paths.

.. currentmodule:: scifem

Mesh utilities
##############

.. autofunction:: create_entity_markers

.. autofunction:: reverse_mark_entities

.. autofunction:: extract_submesh

.. autofunction:: transfer_meshtags_to_submesh

.. autofunction:: find_interface

.. autofunction:: compute_interface_data

.. autofunction:: compute_subdomain_exterior_facets


Function spaces and degrees of freedom
######################################

.. autofunction:: create_real_functionspace

.. autofunction:: create_space_of_simple_functions

.. autofunction:: vertex_to_dofmap

.. autofunction:: dof_to_vertexmap


Assembly
########

.. autofunction:: assemble_scalar

.. autofunction:: norm


Boundary conditions and sources
###############################

.. autofunction:: interpolate_function_onto_facet_dofs

.. autoclass:: PointSource
    :members:
    :undoc-members:


Interpolation
#############

.. autofunction:: interpolation_matrix

.. autofunction:: prepare_interpolation_data

.. autofunction:: petsc_interpolation_matrix


Evaluation and geometry
#######################

.. autofunction:: evaluate_function

.. autofunction:: find_cell_extrema

.. autofunction:: compute_extrema

.. autofunction:: closest_point_projection


Solvers
#######

.. autoclass:: NewtonSolver
    :members:
    :undoc-members:


Periodic meshes
###############

Building a periodic mesh
========================

.. automodule:: scifem.periodic.mesh
    :members:

.. automodule:: scifem.periodic.utils
    :members:

Finding the periodic vertex pairs
=================================

The rebuild above takes the vertex correspondence as input. It can be found from the
coordinates, or read from the ``$Periodic`` section of a gmsh model.

.. automodule:: scifem.periodic.geometrical_search
    :members:

.. automodule:: scifem.periodic.topological_search
    :members:

.. automodule:: scifem.periodic.gmsh
    :members:

Moving data on and off a periodic mesh
======================================

.. automodule:: scifem.periodic.transfer
    :members:

PETSc utilities
###############

.. automodule:: scifem.petsc
    :members:

XDMF output
###########

.. automodule:: scifem.xdmf
    :members:

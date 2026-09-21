#############
API reference
#############

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

Building a periodic mesh, moving data on and off it, and the vertex correspondence the
rebuild runs on -- found either from the coordinates or from the ``$Periodic`` section of
a gmsh model.

.. automodule:: scifem.periodic
    :members:

The two MPI tag constants are documented from the module that defines them, since that is
where their values are written down.

.. autodata:: scifem.periodic.mesh.DEFAULT_TAG_BASE

.. autodata:: scifem.periodic.mesh.NUM_CONSENSUS_TAGS

PETSc utilities
###############

.. automodule:: scifem.petsc
    :members:

XDMF output
###########

.. automodule:: scifem.xdmf
    :members:


Compat functions
################

.. automodule:: scifem.compat
    :members:
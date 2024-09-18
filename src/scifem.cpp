// Copyright (C) 2024 Jørgen S. Dokken and Henrik N.T. Finsberg
// C++ wrappers for SCIFEM

#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

using namespace dolfinx;

namespace scifem
{
    dolfinx::fem::FunctionSpace<double> create_real_functionspace(std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh0)
    {

        basix::FiniteElement e_v = basix::create_element<double>(
            basix::element::family::P, mesh::cell_type_to_basix_type(mesh::CellType::triangle), 0,
            basix::element::lagrange_variant::unset, basix::element::dpc_variant::unset, true);

        // NOTE: Optimize input source/dest later as we know this a priori
        std::int32_t num_dofs = (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0) ? 1 : 0;
        std::int32_t num_ghosts = (dolfinx::MPI::rank(MPI_COMM_WORLD) != 0) ? 1 : 0;
        std::vector<std::int64_t> ghosts(num_ghosts, 0);
        ghosts.reserve(1);
        std::vector<int> owners(num_ghosts, 0);
        owners.reserve(1);
        std::shared_ptr<const dolfinx::common::IndexMap> imap =
            std::make_shared<const dolfinx::common::IndexMap>(MPI_COMM_WORLD, num_dofs, ghosts, owners);
        int index_map_bs = 1;
        int bs = 1;
        // Element dof layout
        fem::ElementDofLayout dof_layout(1, e_v.entity_dofs(), e_v.entity_closure_dofs(), {}, {});
        std::size_t num_cells_on_process = mesh0->topology()->index_map(mesh0->topology()->dim())->size_local() +
                                           mesh0->topology()->index_map(mesh0->topology()->dim())->num_ghosts();

        std::vector<std::int32_t> dofmap(num_cells_on_process, 0);
        dofmap.reserve(1);
        std::shared_ptr<const dolfinx::fem::DofMap> real_dofmap =
            std::make_shared<const dolfinx::fem::DofMap>(dof_layout, imap, index_map_bs, dofmap, bs);
        std::vector<std::size_t> value_shape(0);

        std::shared_ptr<const dolfinx::fem::FiniteElement<double>> d_el =
            std::make_shared<const dolfinx::fem::FiniteElement<double>>(e_v, 1, false);

        return dolfinx::fem::FunctionSpace<double>(mesh0, d_el, real_dofmap, value_shape);
    }
} // namespace scifem
NB_MODULE(_scifem, m)
{
    m.def("create_real_functionspace", &scifem::create_real_functionspace, "Create a real function space");
}
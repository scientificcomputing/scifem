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
#include <memory>
using namespace dolfinx;

namespace scifem
{
    template <typename T>
    dolfinx::fem::FunctionSpace<T> create_real_functionspace(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh)
    {

        basix::FiniteElement e_v = basix::create_element<T>(
            basix::element::family::P, mesh::cell_type_to_basix_type(mesh->topology()->cell_type()), 0,
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
        std::size_t num_cells_on_process = mesh->topology()->index_map(mesh->topology()->dim())->size_local() +
                                           mesh->topology()->index_map(mesh->topology()->dim())->num_ghosts();

        std::vector<std::int32_t> dofmap(num_cells_on_process, 0);
        dofmap.reserve(1);
        std::shared_ptr<const dolfinx::fem::DofMap> real_dofmap =
            std::make_shared<const dolfinx::fem::DofMap>(dof_layout, imap, index_map_bs, dofmap, bs);
        std::vector<std::size_t> value_shape(0);

        std::shared_ptr<const dolfinx::fem::FiniteElement<T>> d_el =
            std::make_shared<const dolfinx::fem::FiniteElement<T>>(e_v, 1, false);

        return dolfinx::fem::FunctionSpace<T>(mesh, d_el, real_dofmap, value_shape);
    }
}

namespace scifem_wrapper
{
    template <typename T>
    void declare_real_function_space(nanobind::module_ &m, std::string type)
    {
        std::string pyfunc_name = "create_real_functionspace_" + type;
        m.def(pyfunc_name.c_str(), [](std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh)
              { return scifem::create_real_functionspace<T>(mesh); }, "Create a real function space");
    }

} // namespace scifem_wrapper

NB_MODULE(_scifem, m)
{
    scifem_wrapper::declare_real_function_space<double>(m, "float64");
    scifem_wrapper::declare_real_function_space<float>(m, "float32");
}
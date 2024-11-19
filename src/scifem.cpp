// Copyright (C) 2024 Jørgen S. Dokken and Henrik N.T. Finsberg
// C++ wrappers for SCIFEM

#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/version.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace
{
// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project (Basix)
// SPDX-License-Identifier:    MIT

template <typename V>
auto as_nbarray(V&& x, std::size_t ndim, const std::size_t* shape)
{
  using _V = std::decay_t<V>;
  _V* ptr = new _V(std::move(x));
  return nanobind::ndarray<typename _V::value_type, nanobind::numpy>(
      ptr->data(), ndim, shape,
      nanobind::capsule(ptr, [](void* p) noexcept { delete (_V*)p; }));
}

template <typename V>
auto as_nbarray(V&& x, const std::initializer_list<std::size_t> shape)
{
  return as_nbarray(x, shape.size(), shape.begin());
}

template <typename V>
auto as_nbarray(V&& x)
{
  return as_nbarray(std::move(x), {x.size()});
}

template <typename V, std::size_t U>
auto as_nbarrayp(std::pair<V, std::array<std::size_t, U>>&& x)
{
  return as_nbarray(std::move(x.first), x.second.size(), x.second.data());
}

} // namespace

namespace scifem
{
template <typename T>
dolfinx::fem::FunctionSpace<T>
create_real_functionspace(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                          std::vector<std::size_t> value_shape)
{

  basix::FiniteElement e_v = basix::create_element<T>(
      basix::element::family::P,
      dolfinx::mesh::cell_type_to_basix_type(mesh->topology()->cell_type()), 0,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, true);

  // NOTE: Optimize input source/dest later as we know this a priori
  std::int32_t num_dofs = (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0) ? 1 : 0;
  std::int32_t num_ghosts = (dolfinx::MPI::rank(MPI_COMM_WORLD) != 0) ? 1 : 0;
  std::vector<std::int64_t> ghosts(num_ghosts, 0);
  ghosts.reserve(1);
  std::vector<int> owners(num_ghosts, 0);
  owners.reserve(1);
  std::shared_ptr<const dolfinx::common::IndexMap> imap
      = std::make_shared<const dolfinx::common::IndexMap>(
          MPI_COMM_WORLD, num_dofs, ghosts, owners);
  std::size_t value_size = std::accumulate(
      value_shape.begin(), value_shape.end(), 1, std::multiplies{});
  int index_map_bs = value_size;
  int bs = value_size;
  // Element dof layout
  dolfinx::fem::ElementDofLayout dof_layout(value_size, e_v.entity_dofs(),
                                            e_v.entity_closure_dofs(), {}, {});
  std::size_t num_cells_on_process
      = mesh->topology()->index_map(mesh->topology()->dim())->size_local()
        + mesh->topology()->index_map(mesh->topology()->dim())->num_ghosts();

  std::vector<std::int32_t> dofmap(num_cells_on_process, 0);
  dofmap.reserve(1);
  std::shared_ptr<const dolfinx::fem::DofMap> real_dofmap
      = std::make_shared<const dolfinx::fem::DofMap>(dof_layout, imap,
                                                     index_map_bs, dofmap, bs);

#if DOLFINX_VERSION_MINOR > 9
  std::shared_ptr<const dolfinx::fem::FiniteElement<T>> d_el
      = std::make_shared<const dolfinx::fem::FiniteElement<T>>(e_v, value_shape,
                                                               false);
  return dolfinx::fem::FunctionSpace<T>(mesh, d_el, real_dofmap);

#else
  std::shared_ptr<const dolfinx::fem::FiniteElement<T>> d_el
      = std::make_shared<const dolfinx::fem::FiniteElement<T>>(e_v, value_size,
                                                               false);
  return dolfinx::fem::FunctionSpace<T>(mesh, d_el, real_dofmap, value_shape);
#endif
}

std::vector<std::int32_t>
create_vertex_to_dofmap(std::shared_ptr<const dolfinx::mesh::Topology> topology,
                        std::shared_ptr<const dolfinx::fem::DofMap> dofmap)
{
  // Get number of vertices
  const std::shared_ptr<const dolfinx::common::IndexMap> v_map
      = topology->index_map(0);
  std::size_t num_vertices = v_map->size_local() + v_map->num_ghosts();

  // Get cell to vertex connectivity
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> c_to_v
      = topology->connectivity(topology->dim(), 0);
  assert(c_to_v);

  // Get vertex dof layout
  const dolfinx::fem::ElementDofLayout& layout = dofmap->element_dof_layout();
  const std::vector<std::vector<std::vector<int>>>& entity_dofs
      = layout.entity_dofs_all();
  const std::vector<std::vector<int>> vertex_dofs = entity_dofs.front();

  // Get number of cells on process
  const std::shared_ptr<const dolfinx::common::IndexMap> c_map
      = topology->index_map(topology->dim());
  std::size_t num_cells = c_map->size_local() + c_map->num_ghosts();

  std::vector<std::int32_t> vertex_to_dofmap(num_vertices);
  for (std::size_t i = 0; i < num_cells; i++)
  {
    auto vertices = c_to_v->links(i);
    auto dofs = dofmap->cell_dofs(i);
    for (std::size_t j = 0; j < vertices.size(); ++j)
    {
      const std::vector<int>& vdof = vertex_dofs[j];
      assert(vdof.size() == 1);
      vertex_to_dofmap[vertices[j]] = dofs[vdof.front()];
    }
  }
  return vertex_to_dofmap;
}

} // namespace scifem

namespace scifem_wrapper
{
template <typename T>
void declare_real_function_space(nanobind::module_& m, std::string type)
{
  std::string pyfunc_name = "create_real_functionspace_" + type;
  m.def(
      pyfunc_name.c_str(),
      [](std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
         std::vector<std::size_t> value_shape)
      { return scifem::create_real_functionspace<T>(mesh, value_shape); },
      "Create a real function space");
  m.def(
      "vertex_to_dofmap",
      [](std::shared_ptr<const dolfinx::mesh::Topology> topology,
         std::shared_ptr<const dolfinx::fem::DofMap> dofmap)
      {
        std::vector<std::int32_t> v_to_d
            = scifem::create_vertex_to_dofmap(topology, dofmap);
        return as_nbarray(v_to_d);
      },
      "Create a vertex to dofmap.");
}

} // namespace scifem_wrapper

NB_MODULE(_scifem, m)
{
  scifem_wrapper::declare_real_function_space<double>(m, "float64");
  scifem_wrapper::declare_real_function_space<float>(m, "float32");
}

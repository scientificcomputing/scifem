// Copyright (C) 2024 Jørgen S. Dokken and Henrik N.T. Finsberg
// C++ wrappers for SCIFEM

#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/version.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
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

template <typename T>
std::tuple<dolfinx::mesh::MeshTags<T>, std::vector<std::int32_t>>
transfer_meshtags_to_submesh(
    const dolfinx::mesh::MeshTags<T>& tags,
    std::shared_ptr<const dolfinx::mesh::Topology> submesh_topology,
    std::span<const std::int32_t> vertex_map,
    std::span<const std::int32_t> cell_map)
{
  int tag_dim = tags.dim();
  int submesh_tdim = submesh_topology->dim();
  auto topology = tags.topology();
  if (tag_dim > submesh_tdim)
  {
    throw std::runtime_error("Tag dimension must be less than or equal to "
                             "submesh dimension");
  }

  std::shared_ptr<const dolfinx::common::IndexMap> parent_sub_entity_map
      = topology->index_map(submesh_tdim);
  if (!parent_sub_entity_map)
  {
    throw std::runtime_error("Parent entities of dimension "
                             + std::to_string(submesh_tdim)
                             + " not found in topology");
  }

  // Invert submap cell to parent entity map
  std::int32_t parent_num_sub_entities = parent_sub_entity_map->size_local()
                                         + parent_sub_entity_map->num_ghosts();
  std::vector<std::int32_t> parent_entity_to_sub_cell(parent_num_sub_entities,
                                                      -1);
  for (std::int32_t k = 0; k < cell_map.size(); ++k)
    parent_entity_to_sub_cell[cell_map[k]] = k;

  // Access various connectivity maps
  auto sub_e_to_v = submesh_topology->connectivity(tag_dim, 0);
  if (!sub_e_to_v)
  {
    throw std::runtime_error("Missing submesh connectivity between "
                             + std::to_string(tag_dim) + " and 0");
  }
  auto sub_c_to_e = submesh_topology->connectivity(submesh_tdim, tag_dim);
  if (!sub_c_to_e)
  {
    throw std::runtime_error("Missing submesh connectivity between "
                             + std::to_string(submesh_tdim) + " and "
                             + std::to_string(tag_dim));
  }
  auto e_to_v = topology->connectivity(tag_dim, 0);
  if (!e_to_v)
  {
    throw std::runtime_error("Missing connectivity between "
                             + std::to_string(tag_dim) + " and 0");
  }
  auto e_to_sub_cell = topology->connectivity(tag_dim, submesh_tdim);
  if (!e_to_sub_cell)
  {
    throw std::runtime_error("Missing connectivity between "
                             + std::to_string(tag_dim) + " and "
                             + std::to_string(submesh_tdim));
  }

  auto sub_entity_map = submesh_topology->index_map(tag_dim);

  // Prepare sub entity to parent map
  std::size_t num_sum_entities
      = sub_entity_map->size_local() + sub_entity_map->num_ghosts();
  std::vector<std::int32_t> sub_entity_to_parent(num_sum_entities, -1);

  // Initialize submesh values with numerical min
  std::vector<T> submesh_values(num_sum_entities,
                                std::numeric_limits<T>::min());
  std::vector<std::int32_t> submesh_indices(num_sum_entities);
  std::iota(submesh_indices.begin(), submesh_indices.end(), 0);

  // Map tag indices to global index
  std::span<const std::int32_t> tag_indices = tags.indices();
  auto parent_entity_map = topology->index_map(tag_dim);
  std::vector<std::int64_t> global_tag_indices(tag_indices.size());
  parent_entity_map->local_to_global(tag_indices, global_tag_indices);

  // Accumulate global indices across processes
  dolfinx::la::Vector<std::int64_t> index_mapper(parent_entity_map, 1);
  index_mapper.set(-1);
  std::span<std::int64_t> indices = index_mapper.mutable_array();
  for (std::size_t i = 0; i < global_tag_indices.size(); ++i)
    indices[tag_indices[i]] = global_tag_indices[i];
  index_mapper.scatter_rev([](std::int32_t a, std::int32_t b)
                           { return std::max<std::int32_t>(a, b); });
  index_mapper.scatter_fwd();

  // Map tag values in a similar way (Allowing negative values)
  dolfinx::la::Vector<T> values_mapper(parent_entity_map, 1);
  values_mapper.set(std::numeric_limits<T>::min());
  std::span<T> values = values_mapper.mutable_array();
  std::span<const T> tag_values = tags.values();
  for (std::size_t i = 0; i < tag_values.size(); ++i)
    values[tag_indices[i]] = tag_values[i];
  values_mapper.scatter_rev([](T a, T b) { return std::max<T>(a, b); });
  values_mapper.scatter_fwd();

  // For each entity in the tag, find all cells of the submesh connected to this
  // entity. Global to local returns -1 if not on process.
  std::vector<std::int32_t> local_indices(indices.size());
  parent_entity_map->global_to_local(indices, local_indices);
  std::span<const T> parent_values = values_mapper.array();

  if (local_indices.size() != parent_values.size())
    throw std::runtime_error("Number of indices and values do not match");

  for (std::size_t i = 0; i < local_indices.size(); ++i)
  {

    auto parent_entity = local_indices[i];
    auto parent_value = parent_values[i];
    if (parent_entity == -1)
      continue;
    assert(parent_entity < e_to_v->num_nodes());

    auto entity_vertices = e_to_v->links(parent_entity);
    bool entity_found = false;
    for (auto parent_cell : e_to_sub_cell->links(parent_entity))
    {
      if (entity_found)
        break;

      if (std::int32_t sub_cell = parent_entity_to_sub_cell[parent_cell];
          sub_cell > -1)
      {

        // For a cell in the sub mesh find all attached entities,
        // and define them by their vertices in the sub mesh
        assert(sub_cell < sub_c_to_e->num_nodes());
        for (auto sub_entity : sub_c_to_e->links(sub_cell))
        {
          if (entity_found)
            break;
          auto sub_vertices = sub_e_to_v->links(sub_entity);
          bool entity_matches = true;
          for (auto sub_vertex : sub_vertices)
          {
            if (std::find(entity_vertices.begin(), entity_vertices.end(),
                          vertex_map[sub_vertex])
                == entity_vertices.end())
            {
              entity_matches = false;
              break;
            }
          }
          if (entity_matches)
          {
            // Found entity in submesh with the same vertices as in the
            // parent mesh
            submesh_values[sub_entity] = parent_value;
            entity_found = true;
            sub_entity_to_parent[sub_entity] = parent_entity;
          }
        }
      }
    }
  }
  dolfinx::mesh::MeshTags<T> new_meshtag(submesh_topology, tag_dim,
                                         submesh_indices, submesh_values);
  return std::make_tuple(new_meshtag, sub_entity_to_parent);
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

template <typename T>
void declare_meshtag_operators(nanobind::module_& m, std::string type)
{
  std::string pyfunc_name = "transfer_meshtags_to_submesh_" + type;
  m.def(
      pyfunc_name.c_str(),
      [](const dolfinx::mesh::MeshTags<T>& tags,
         std::shared_ptr<const dolfinx::mesh::Topology> submesh_topology,
         nanobind::ndarray<const std::int32_t, nanobind::ndim<1>,
                           nanobind::c_contig>
             vertex_map,
         nanobind::ndarray<const std::int32_t, nanobind::ndim<1>,
                           nanobind::c_contig>
             cell_map)
      {
        std::tuple<dolfinx::mesh::MeshTags<T>, std::vector<std::int32_t>>
            sub_data = scifem::transfer_meshtags_to_submesh<T>(
                tags, submesh_topology,
                std::span(vertex_map.data(), vertex_map.size()),
                std::span(cell_map.data(), cell_map.size()));
        auto _e_map = as_nbarray(std::move(std::get<1>(sub_data)));
        return std::tuple(std::move(std::get<0>(sub_data)), _e_map);
      },
      nanobind::arg("tags"), nanobind::arg("submesh_topology"),
      nanobind::arg("vertex_map"), nanobind::arg("cell_map"));
}

} // namespace scifem_wrapper

NB_MODULE(_scifem, m)
{
  scifem_wrapper::declare_real_function_space<double>(m, "float64");
  scifem_wrapper::declare_real_function_space<float>(m, "float32");
  scifem_wrapper::declare_meshtag_operators<std::int32_t>(m, "int32");
}

// Copyright (C) 2024-2026 Jørgen S. Dokken and Henrik N.T. Finsberg
// C++ wrappers for SCIFEM

#include <algorithm>
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/version.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <iostream>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <numeric>
#include <thread>

namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

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

namespace impl
{

template <typename T>
inline void sort_array_3_descending(std::array<T, 3>& arr)
{
  // If the second element is larger, swap it to the front
  if (arr[1] > arr[0])
    std::swap(arr[0], arr[1]);

  // If the third element is larger, swap it up
  if (arr[2] > arr[1])
    std::swap(arr[1], arr[2]);

  // One final check between the first two elements
  if (arr[1] > arr[0])
    std::swap(arr[0], arr[1]);
}

template <typename T>
inline void sort_array_2_descending(std::array<T, 2>& arr)
{
  if (arr[1] > arr[0])
    std::swap(arr[0], arr[1]);
}

template <typename T, std::size_t tdim>
inline void simplex_projection(const std::array<T, tdim>& x,
                               std::array<T, tdim>& projected)
{
  std::array<T, tdim> u;
  std::ranges::transform(x, u.begin(),
                         [](auto xi) { return std::max(xi, T(0)); });
  T sum = std::accumulate(u.begin(), u.end(), T(0));
  if (sum <= (T)1.0)
  {
    if constexpr (tdim == 2)
    {
      projected[0] = u[0];
      projected[1] = u[1];
      return;
    }
    else if constexpr (tdim == 3)
    {
      projected[0] = u[0];
      projected[1] = u[1];
      projected[2] = u[2];
      return;
    }
    else
    {
      static_assert(tdim == 2 || tdim == 3, "Unsupported dimension");
    }
  }
  std::array<T, tdim> x_sorted = x;
  std::array<T, tdim> cumsum;
  std::array<bool, tdim> Ks;
  std::size_t K;
  if constexpr (tdim == 2)
  {
    impl::sort_array_2_descending(x_sorted);
    cumsum[0] = x_sorted[0];
    cumsum[1] = x_sorted[0] + x_sorted[1];
    Ks[0] = x_sorted[0] > (cumsum[0] - (T)1.0);
    Ks[1] = x_sorted[1] * (T)2 > (cumsum[1] - (T)1.0);
    if (Ks[1])
      K = 2;
    else if (Ks[0])
      K = 1;
    else
      throw std::runtime_error("Unexpected condition in simplex projection.");
  }
  else if constexpr (tdim == 3)
  {
    impl::sort_array_3_descending(x_sorted);
    cumsum[0] = x_sorted[0];
    cumsum[1] = x_sorted[0] + x_sorted[1];
    cumsum[2] = x_sorted[0] + x_sorted[1] + x_sorted[2];
    Ks[0] = x_sorted[0] > (cumsum[0] - (T)1.0);
    Ks[1] = x_sorted[1] * (T)2 > (cumsum[1] - (T)1.0);
    Ks[2] = x_sorted[2] * (T)3 > (cumsum[2] - (T)1.0);
    if (Ks[2])
      K = 3;
    else if (Ks[1])
      K = 2;
    else if (Ks[0])
      K = 1;
    else
      throw std::runtime_error("Unexpected condition in simplex projection.");
  }
  else
  {
    static_assert(tdim == 2 || tdim == 3, "Unsupported dimension");
  }
  T tau = (cumsum[K - 1] - (T)1.0) / T(K);
  for (std::size_t i = 0; i < tdim; ++i)
    projected[i] = std::max(x[i] - tau, T(0));
};

template <typename T, std::size_t tdim, bool is_simplex>
inline void projection(const std::array<T, tdim>& x,
                       std::array<T, tdim>& projected)
{
  if constexpr (is_simplex)
    simplex_projection(x, projected);
  else
    for (std::size_t i = 0; i < tdim; ++i)
      projected[i] = std::clamp(x[i], T(0.0), T(1.0));
}

} // namespace impl

namespace scifem
{

template <typename T, std::size_t tdim, bool is_simplex, std::size_t gdim>
std::tuple<std::vector<T>, std::vector<T>> closest_point_projection(
    const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::int32_t> cells,
    std::span<const T> points, T tol_x, T tol_dist, T tol_grad,
    std::size_t max_iter, std::size_t max_ls_iter, std::size_t num_threads)
{
  constexpr T eps = std::numeric_limits<T>::epsilon();
  constexpr T roundoff_tol = 100 * eps;

  const dolfinx::fem::CoordinateElement<T>& cmap = mesh.geometry().cmap();
  std::vector<T> closest_points(3 * cells.size(), T(0));
  assert(cells.size() == points.size() / 3);
  std::vector<T> reference_points(cells.size() * tdim);

  std::array<T, tdim> initial_guess;
  T midpoint_divider = (is_simplex) ? 1.0 : 0.0;
  std::ranges::fill(initial_guess, 1.0 / ((T)tdim + midpoint_divider));

  const std::array<std::size_t, 4> dtab_shape = cmap.tabulate_shape(1, 1);
  const std::array<std::size_t, 4> tab_shape = cmap.tabulate_shape(0, 1);
  const std::size_t basis_data_size
      = std::reduce(tab_shape.begin(), tab_shape.end(), 1, std::multiplies{});

  md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> x_dofmap
      = mesh.geometry().dofmap(0);
  std::span<const T> x = mesh.geometry().x();

  auto compute_chunk = [&](std::size_t c0, std::size_t c1)
  {
    // Use single buffer for 0th and 1st order derivatives, as we only need one
    // at a time
    std::vector<T> dphi_b(std::reduce(dtab_shape.begin(), dtab_shape.end(), 1,
                                      std::multiplies{}));
    md::mdspan<const T, md::dextents<std::size_t, 4>> phi_full(dphi_b.data(),
                                                               dtab_shape);
    auto phi = md::submdspan(phi_full, 0, md::full_extent, md::full_extent, 0);
    auto dphi = md::submdspan(phi_full, std::pair(1, tdim + 1), 0,
                              md::full_extent, 0);

    std::array<T, tdim> x_k;
    std::array<T, tdim> x_old;
    std::array<T, gdim> diff;
    std::array<T, gdim * tdim> J_buffer;
    std::array<T, tdim> gradient;
    std::array<T, tdim> x_new_prev;
    std::array<T, tdim> x_new;
    std::array<T, tdim> x_k_tmp;
    std::array<T, gdim> target_point;

    md::mdspan<T, md::extents<std::size_t, gdim, tdim>> J(J_buffer.data(), gdim,
                                                          tdim);

    std::array<T, gdim> X_phys_buffer;
    md::mdspan<T, md::extents<std::size_t, 1, gdim>> surface_point(
        X_phys_buffer.data(), 1, gdim);

    // Extract data to compute cell geometry
    std::vector<T> cdofs(3 * x_dofmap.extent(1));
    md::mdspan<const T, md::dextents<std::size_t, 2>> cell_geometry(
        cdofs.data(), x_dofmap.extent(1), gdim);

    constexpr T sigma = 0.1;
    constexpr T beta = 0.5;

    for (std::size_t i = c0; i < c1; ++i)
    {
      // Pack cell geometry into mdspan for push_forward
      auto x_dofs = md::submdspan(x_dofmap, cells[i], md::full_extent);
      for (std::size_t k = 0; k < x_dofs.size(); ++k)
        std::copy_n(x.data() + (3 * x_dofs[k]), gdim,
                    std::next(cdofs.begin(), gdim * k));

      std::copy(points.data() + (3 * i), points.data() + (3 * i) + gdim,
                target_point.begin());

      // Update Initial guess
      std::ranges::copy(initial_guess, x_k.begin());
      for (std::size_t k = 0; k < max_iter; k++)
      {
        for (std::size_t l = 0; l < tdim; ++l)
          x_old[l] = x_k[l];

        // Tabulate basis function and gradient at current point
        std::ranges::fill(dphi_b, 0);
        cmap.tabulate(1, std::span(x_k.data(), tdim), {1, tdim}, dphi_b);

        // Push forward to physicalspace
        dolfinx::fem::CoordinateElement<T>::push_forward(surface_point,
                                                         cell_geometry, phi);

        // Compute objective function (squared distance to point)/2
        T current_dist_sq = 0;
        for (std::size_t d = 0; d < gdim; ++d)
        {
          diff[d] = X_phys_buffer[d] - target_point[d];
          current_dist_sq += diff[d] * diff[d];
        }
        current_dist_sq *= T(0.5);

        // Compute Jacobian (tangent vectors)
        for (std::size_t l = 0; l < tdim * gdim; l++)
          J_buffer[l] = 0;
        dolfinx::fem::CoordinateElement<T>::compute_jacobian(dphi,
                                                             cell_geometry, J);

        // Compute tangents (J^T diff)
        for (std::size_t m = 0; m < tdim; ++m)
          gradient[m] = 0;
        for (std::size_t l = 0; l < gdim; ++l)
          for (std::size_t m = 0; m < tdim; ++m)
            gradient[m] += J(l, m) * diff[l];

        // Check for convergence in gradient norm, scaled by Jacobian to account
        // for stretching of the reference space
        T jac_norm = 0;
        for (std::size_t l = 0; l < tdim * gdim; l++)
          jac_norm += J_buffer[l] * J_buffer[l];
        T scaled_tol_grad = tol_grad * std::max(jac_norm, T(1));
        T g_squared = 0;
        for (std::size_t l = 0; l < tdim; ++l)
          g_squared += gradient[l] * gradient[l];
        if (g_squared < scaled_tol_grad)
        {
          break;
        }

        // Goldstein-Polyak-Levitin Projected Line Search
        // Bertsekas (1976) Eq. (14) - Armijo Rule along the Projection Arc
        for (std::size_t l = 0; l < tdim; ++l)
          x_new_prev[l] = -1;
        bool target_reached = false;
        T alpha = 1.0;
        for (std::size_t ls_iter = 0; ls_iter < max_ls_iter; ++ls_iter)
        {
          // Take projected gradient step
          // Compute new step xk - alpha * g and project back to domain
          for (std::size_t l = 0; l < tdim; l++)
            x_k_tmp[l] = x_k[l] - alpha * gradient[l];
          impl::projection<T, tdim, is_simplex>(x_k_tmp, x_new);

          // The projection is pinned to the boundary, terminate search
          T local_diff = 0;
          for (std::size_t l = 0; l < tdim; l++)
            local_diff
                += (x_new[l] - x_new_prev[l]) * (x_new[l] - x_new_prev[l]);
          if (local_diff < eps * eps)
            break;
          for (std::size_t l = 0; l < tdim; l++)
            x_new_prev[l] = x_new[l];

          // Evaluate distance at new point
          std::ranges::fill(std::span(dphi_b.data(), basis_data_size), T(0));
          cmap.tabulate(0, std::span(x_new.data(), tdim), {1, tdim},
                        std::span(dphi_b.data(), basis_data_size));

          // Push forward to physical space
          dolfinx::fem::CoordinateElement<T>::push_forward(surface_point,
                                                           cell_geometry, phi);

          // Compute objective function (squared distance to point)/2
          T new_sq_dist = 0;
          for (std::size_t l = 0; l < gdim; ++l)
          {
            diff[l] = X_phys_buffer[l] - target_point[l];
            new_sq_dist += diff[l] * diff[l];
          }
          new_sq_dist *= 0.5;

          // If we are close enough to the targetpoint we terminate the
          // linesearch, even if the Armijo condition is not satisfied, to avoid
          // unnecessary iterations close to the target point.
          if (new_sq_dist < 0.5 * tol_dist * tol_dist)
          {
            target_reached = true;
            break;
          }

          // Bertsekas Eq. (14) condition:
          // f(x_new) <= f(x_k) + sigma * grad_f(x_k)^T * (x_new - x_k)
          // Note: g is grad_f(x_k)
          // Size of step after projecting back to boundary
          T grad_dot_step = 0;
          for (std::size_t d = 0; d < tdim; ++d)
            grad_dot_step += gradient[d] * (x_new[d] - x_k[d]);
          if (new_sq_dist
              <= current_dist_sq + sigma * grad_dot_step + roundoff_tol)
          {
            break;
          }

          alpha *= beta;
          if (ls_iter == max_ls_iter - 1)
          {
            std::string out_msg = std::format(
                "Line search failed to find a suitable step after {} "
                "iterations. "
                "Current dist: {}, grad_dot_step: {}, alpha: {}",
                max_ls_iter, new_sq_dist, grad_dot_step, alpha);
            std::cout << out_msg << std::endl;
          }
        }

        std::ranges::copy(x_new, x_k.begin());
        if (target_reached)
          break;

        // Check if Newton iteration has converged
        T x_diff_sq = 0;
        for (std::size_t l = 0; l < tdim; l++)
          x_diff_sq += (x_k[l] - x_old[l]) * (x_k[l] - x_old[l]);
        if (x_diff_sq < tol_x * tol_x)
          break;

        if (k == max_iter - 1)
        {
          std::string error_msg
              = std::format("Newton iteration failed to converge after {} "
                            "iterations. x_diff_sq={} J={}",
                            max_iter, x_diff_sq, current_dist_sq);
          throw std::runtime_error(error_msg);
        }
      }
      // Push forward to physicalspace for closest point
      std::ranges::fill(std::span(dphi_b.data(), basis_data_size), T(0));
      cmap.tabulate(0, std::span(x_k.data(), tdim), {1, tdim},
                    std::span(dphi_b.data(), basis_data_size));
      dolfinx::fem::CoordinateElement<T>::push_forward(surface_point,
                                                       cell_geometry, phi);

      std::copy_n(X_phys_buffer.data(), gdim,
                  std::next(closest_points.begin(), 3 * i));
      std::copy_n(x_k.begin(), tdim,
                  std::next(reference_points.begin(), tdim * i));
    }
  };

  size_t total_cells = cells.size();
  num_threads = std::max<size_t>(1, std::min(num_threads, total_cells));
  // --- THREAD EXECUTION ---
  if (num_threads <= 1)
  {
    compute_chunk(0, total_cells);
  }
  else
  {
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
    {
      auto [c0, c1] = dolfinx::MPI::local_range(i, total_cells, num_threads);
      threads.emplace_back(compute_chunk, c0, c1);
    }
  }
  return {std::move(closest_points), std::move(reference_points)};
}

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

  // We select the process that owns cell 0 to have all dofs and all other
  // processes ghost them
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = mesh->topology()->index_map(mesh->topology()->dim());
  const int is_owner
      = (cell_map->local_range()[0] == 0) and cell_map->size_local() > 0;

  std::int32_t num_dofs = (is_owner) ? 1 : 0;
  std::int32_t num_ghosts = (is_owner) ? 0 : 1;
  std::vector<std::int64_t> ghosts(num_ghosts, 0);
  ghosts.reserve(1);

  int rank = dolfinx::MPI::rank(mesh->comm());
  std::array<int, 2> send_owner_pair = {is_owner, rank};
  std::array<int, 2> recv_owner_pair;
  MPI_Allreduce(&send_owner_pair, &recv_owner_pair, 1, MPI_2INT, MPI_MAXLOC,
                mesh->comm());
  std::vector<int> owners(num_ghosts, recv_owner_pair[1]);
  owners.reserve(1);
  std::shared_ptr<const dolfinx::common::IndexMap> imap
      = std::make_shared<const dolfinx::common::IndexMap>(
          mesh->comm(), num_dofs, ghosts, owners);
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
  std::ranges::fill(index_mapper.array(), -1);
  std::span<std::int64_t> indices = index_mapper.array();
  for (std::size_t i = 0; i < global_tag_indices.size(); ++i)
    indices[tag_indices[i]] = global_tag_indices[i];
  index_mapper.scatter_rev([](std::int32_t a, std::int32_t b)
                           { return std::max<std::int32_t>(a, b); });
  index_mapper.scatter_fwd();

  // Map tag values in a similar way (Allowing negative values)
  dolfinx::la::Vector<T> values_mapper(parent_entity_map, 1);
  std::ranges::fill(values_mapper.array(), std::numeric_limits<T>::min());
  std::span<T> values = values_mapper.array();
  std::span<const T> tag_values = tags.values();
  for (std::size_t i = 0; i < tag_values.size(); ++i)
    values[tag_indices[i]] = tag_values[i];
  values_mapper.scatter_rev([](T a, T b) { return std::max<T>(a, b); });
  values_mapper.scatter_fwd();

  // For each entity in the tag, find all cells of the submesh connected to
  // this entity. Global to local returns -1 if not on process.
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

template <typename T>
void declare_closest_point(nanobind::module_& m, std::string type)
{
  std::string pyfunc_name = "closest_point_projection_" + type;
  m.def(
      pyfunc_name.c_str(),
      [](const dolfinx::mesh::Mesh<T>& mesh,
         nanobind::ndarray<const std::int32_t, nanobind::ndim<1>,
                           nanobind::c_contig>
             cells,
         nanobind::ndarray<const T, nanobind::ndim<2>, nanobind::c_contig>
             points,
         T tol_x, T tol_dist, T tol_grad, std::size_t max_iter,
         std::size_t max_ls_iter, std::size_t num_threads)
      {
        std::size_t tdim
            = dolfinx::mesh::cell_dim(mesh.topology()->cell_type());
        std::size_t gdim = mesh.geometry().dim();
        bool is_simplex
            = (mesh.topology()->cell_type() == dolfinx::mesh::CellType::triangle
               || mesh.topology()->cell_type()
                      == dolfinx::mesh::CellType::tetrahedron);
        switch (tdim)
        {
        case 1:
        {
          switch (gdim)
          {
          case 1:
          {
            auto [closest_point, closest_ref]
                = scifem::closest_point_projection<T, 1, false, 1>(
                    mesh,
                    std::span<const std::int32_t>(cells.data(), cells.size()),
                    std::span<const T>(points.data(), points.size()), tol_x,
                    tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
            return std::make_tuple(
                as_nbarray(closest_point, {cells.size(), 3}),
                as_nbarray(closest_ref, {cells.size(), tdim}));
          }
          case 2:
          {
            auto [closest_point, closest_ref]
                = scifem::closest_point_projection<T, 1, false, 2>(
                    mesh,
                    std::span<const std::int32_t>(cells.data(), cells.size()),
                    std::span<const T>(points.data(), points.size()), tol_x,
                    tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
            return std::make_tuple(
                as_nbarray(closest_point, {cells.size(), 3}),
                as_nbarray(closest_ref, {cells.size(), tdim}));
          }
          case 3:
          {
            auto [closest_point, closest_ref]
                = scifem::closest_point_projection<T, 1, false, 3>(
                    mesh,
                    std::span<const std::int32_t>(cells.data(), cells.size()),
                    std::span<const T>(points.data(), points.size()), tol_x,
                    tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
            return std::make_tuple(
                as_nbarray(closest_point, {cells.size(), 3}),
                as_nbarray(closest_ref, {cells.size(), tdim}));
          };

          default:
            throw std::runtime_error("Unsupported geometric dimension");
          }
        }
        case 2:
        {
          if (is_simplex)
          {
            switch (gdim)
            {
            case 2:
            {
              auto [closest_point, closest_ref]
                  = scifem::closest_point_projection<T, 2, true, 2>(
                      mesh,
                      std::span<const std::int32_t>(cells.data(), cells.size()),
                      std::span<const T>(points.data(), points.size()), tol_x,
                      tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
              return std::make_tuple(
                  as_nbarray(closest_point, {cells.size(), 3}),
                  as_nbarray(closest_ref, {cells.size(), tdim}));
            }
            case 3:
            {
              auto [closest_point, closest_ref]
                  = scifem::closest_point_projection<T, 2, true, 3>(
                      mesh,
                      std::span<const std::int32_t>(cells.data(), cells.size()),
                      std::span<const T>(points.data(), points.size()), tol_x,
                      tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
              return std::make_tuple(
                  as_nbarray(closest_point, {cells.size(), 3}),
                  as_nbarray(closest_ref, {cells.size(), tdim}));
            }
            default:
              throw std::runtime_error("Unsupported geometric dimension");
            }
          }
          else
          {
            switch (gdim)
            {
            case 2:
            {
              auto [closest_point, closest_ref]
                  = scifem::closest_point_projection<T, 2, false, 2>(
                      mesh,
                      std::span<const std::int32_t>(cells.data(), cells.size()),
                      std::span<const T>(points.data(), points.size()), tol_x,
                      tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
              return std::make_tuple(
                  as_nbarray(closest_point, {cells.size(), 3}),
                  as_nbarray(closest_ref, {cells.size(), tdim}));
            }
            case 3:
            {
              auto [closest_point, closest_ref]
                  = scifem::closest_point_projection<T, 2, false, 3>(
                      mesh,
                      std::span<const std::int32_t>(cells.data(), cells.size()),
                      std::span<const T>(points.data(), points.size()), tol_x,
                      tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
              return std::make_tuple(
                  as_nbarray(closest_point, {cells.size(), 3}),
                  as_nbarray(closest_ref, {cells.size(), tdim}));
            }
            default:
              throw std::runtime_error("Unsupported geometric dimension");
            }
          }
        }
        case 3:
        {
          if (is_simplex)
          {
            auto [closest_point, closest_ref]
                = scifem::closest_point_projection<T, 3, true, 3>(
                    mesh,
                    std::span<const std::int32_t>(cells.data(), cells.size()),
                    std::span<const T>(points.data(), points.size()), tol_x,
                    tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
            return std::make_tuple(
                as_nbarray(closest_point, {cells.size(), 3}),
                as_nbarray(closest_ref, {cells.size(), tdim}));
          }
          else
          {
            auto [closest_point, closest_ref]
                = scifem::closest_point_projection<T, 3, false, 3>(
                    mesh,
                    std::span<const std::int32_t>(cells.data(), cells.size()),
                    std::span<const T>(points.data(), points.size()), tol_x,
                    tol_dist, tol_grad, max_iter, max_ls_iter, num_threads);
            return std::make_tuple(
                as_nbarray(closest_point, {cells.size(), 3}),
                as_nbarray(closest_ref, {cells.size(), tdim}));
          }

        default:
          throw std::runtime_error("Unsupported cell dimension");
        }
        }
      },
      nanobind::arg("mesh"), nanobind::arg("cells"), nanobind::arg("points"),
      nanobind::arg("tol_x"), nanobind::arg("tol_dist"),
      nanobind::arg("tol_grad"), nanobind::arg("max_iter"),
      nanobind::arg("max_ls_iter"), nanobind::arg("num_threads"),
      "Compute the closest points on the mesh to a set of input points, and "
      "return the closest points and their reference coordinates.");
}

} // namespace scifem_wrapper

NB_MODULE(_scifem, m)
{
  scifem_wrapper::declare_real_function_space<double>(m, "float64");
  scifem_wrapper::declare_real_function_space<float>(m, "float32");
  scifem_wrapper::declare_meshtag_operators<std::int32_t>(m, "int32");
  scifem_wrapper::declare_closest_point<double>(m, "float64");
  scifem_wrapper::declare_closest_point<float>(m, "float32");
}

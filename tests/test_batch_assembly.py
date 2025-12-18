# NO VIBE CODING

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
import ufl
import numpy as np
import time

def extract_integration_domains(form: ufl.Form)->tuple[ufl.Form, dict[dolfinx.fem.IntegralType, list[tuple[int, np.ndarray]]]]:
    """Extract integration domains from form and replace everywhere integrals with a unique tag index."""

    integrals = form.integrals()
    new_integrals = []

    # Given old form, extract tagged entities at this point   
    all_itg_ids = set()
    integral_types = set()
    for integral in integrals:
        itg_type = integral.integral_type()
        integral_types.add(itg_type)
        itg_id = integral.subdomain_id()
        if itg_id != "everywhere":
            if isinstance(itg_id, int):
                all_itg_ids.add(itg_id)
            else:
                for id in itg_id:
                    all_itg_ids.add(id)

    # Repack integration data manually for each integrand
    # NOTE: This tag has to be unique per integration domain if we have submeshes with measure of same cell type
    everywhere_tag = max(all_itg_ids) + 1 if len(all_itg_ids) > 0 else 1

    new_integrals = []
    integrals_ids_by_type: dict[tuple[ufl.Mesh, str], set[int]] = {}
    for integral in integrals:
        itg_type = integral.integral_type()
        if (itg_id := integral.subdomain_id()) == "everywhere":
            itg_id = everywhere_tag
        key = (integral.ufl_domain(), itg_type)
        if integrals_ids_by_type.get(key, None) is None:
            integrals_ids_by_type[key] = set()

        if isinstance(itg_id, int):
            integrals_ids_by_type[key].add(itg_id)
        else:
            for iid in itg_id:
                integrals_ids_by_type[key].add(iid)
        new_integrals.append(integral.reconstruct(subdomain_id=itg_id))
    new_form = ufl.Form(new_integrals)
    subdomain_data = F.subdomain_data()

    integral_data: dict[dolfinx.fem.IntegralType, list[tuple[int, np.ndarray]]] = {}
    string_to_type = {itg.name: itg for itg in dolfinx.fem.IntegralType}
    type_to_codim = {dolfinx.fem.IntegralType.cell: 0,
                     dolfinx.fem.IntegralType.exterior_facet: 1,
                     dolfinx.fem.IntegralType.interior_facet: 1,
                     dolfinx.fem.IntegralType.ridge: 2}
    for key, ids in integrals_ids_by_type.items():
        ufl_domain, itg_type = key
        tags = set(filter(lambda data: data is not None, subdomain_data[ufl_domain][itg_type]))
        dfx_type = string_to_type[itg_type]
        if integral_data.get(dfx_type, None) is None:
            integral_data[dfx_type] = []

        if len(tags) == 1:
            tag = list(tags)[0]
            for itg_id in ids:
                if itg_id == everywhere_tag:
                    edim = tag.topology.dim - type_to_codim[dfx_type]
                    if dfx_type ==dolfinx.fem.IntegralType.exterior_facet:
                        entities = dolfinx.cpp.mesh.exterior_facet_indices(tag.topology)
                    else:
                        num_entities_local = tag.topology.index_map(edim).size_local
                        entities = np.arange(num_entities_local, dtype=np.int32)
                    integration_entities = dolfinx.cpp.fem.compute_integration_domains(dfx_type, tag.topology, entities)
                else:
                    
                    integration_entities = dolfinx.cpp.fem.compute_integration_domains(dfx_type, tag.topology, tag.find(itg_id))
                integral_data[dfx_type].append((itg_id, integration_entities))
        else:
            assert len(ids) == 1 and list(ids)[0] == everywhere_tag
            topology = ufl_domain.ufl_cargo().topology
            edim = topology.dim - type_to_codim[dfx_type]
            if dfx_type ==dolfinx.fem.IntegralType.exterior_facet:
                entities = dolfinx.mesh.exterior_facet_indices(mesh.topology)
            else:
                num_entities_local = topology.index_map(edim).size_local
                entities = np.arange(num_entities_local)
            integration_entities = dolfinx.cpp.fem.compute_integration_domains(dfx_type, topology, entities)
            integral_data[dfx_type].append((everywhere_tag, integration_entities))
    return new_form, integral_data

def compute_stride_distribution(M, max_batches):
    stride_offset = np.zeros(max_batches + 1, dtype=np.int32)
    if M == 0:
        return stride_offset
    data_per_stride = np.zeros(max_batches)

    data_per_batch = M // max_batches
    remainders = M % max_batches
    data_per_stride[:max_batches] = data_per_batch
    data_per_stride[:remainders] += 1
    assert int(np.sum(data_per_stride)) == M
    stride_offset[1:] = np.cumsum(data_per_stride)
    return stride_offset.astype(np.int32)

def create_idata_batches(idata: dict[dolfinx.fem.IntegralType, list[tuple[int, np.ndarray]]], num_batches:int = 10) -> list[dict[dolfinx.fem.IntegralType, list[tuple[int, np.ndarray]]]]:
    batched_integral_data = [{itg_type: [] for itg_type in idata.keys()} for _ in range(num_batches)]
    estride = {dolfinx.fem.IntegralType.cell: 1,
               dolfinx.fem.IntegralType.exterior_facet: 2,
               dolfinx.fem.IntegralType.interior_facet: 4,
               dolfinx.fem.IntegralType.ridge: 2,
               dolfinx.fem.IntegralType.vertex: 2}
    for itg_type, integration_entities in idata.items():
        for (tag, itg_entities) in integration_entities:
            assert len(itg_entities)% estride[itg_type] == 0
            num_entities = len(itg_entities)//estride[itg_type]
            stride_dist = compute_stride_distribution(num_entities, num_batches)*int(estride[itg_type])
            for i in range(num_batches):
                batched_integral_data[i][itg_type].append((tag, itg_entities[stride_dist[i]:stride_dist[i+1]]))
    return batched_integral_data


def create_batched_form(F: ufl.Form, num_batches: int, entity_maps:list[dolfinx.mesh.EntityMap]=None) -> list[dolfinx.fem.Form]:
    new_form, integral_data = extract_integration_domains(F)
    batched_integral_data = create_idata_batches(integral_data, num_batches=num_batches)
    function_spaces = [arg.ufl_function_space() for arg in new_form.arguments()]
    coefficient_map = {coeff:coeff for coeff in new_form.coefficients()}
    constant_map = {const: const for const in new_form.constants()}
    compiled_form = dolfinx.fem.compile_form(MPI.COMM_WORLD, new_form)

    F_batched = [dolfinx.fem.create_form(compiled_form, function_spaces=function_spaces,
                                      msh=mesh, subdomains=batched_data,
                                      coefficient_map=coefficient_map, constant_map=constant_map,
                                      entity_maps=entity_maps) for batched_data in batched_integral_data
]

    return F_batched

if __name__ == "__main__":
    M = 400
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 200, 200)
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), np.ones(num_cells_local, dtype=np.int32))
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 8))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1]**2)

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    num_facets_local = mesh.topology.index_map(mesh.topology.dim-1).size_local
    def marker(x):
        return x[0]<x[1]
    facets = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim-1, marker)
    ft = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, facets, np.ones_like(facets))


    v = ufl.TestFunction(V)

    F= ufl.inner(u, v) * ufl.dx(subdomain_data=ct, subdomain_id=4)\
    +ufl.inner(u, v) * ufl.dx \
        +ufl.inner(u, v) * ufl.ds \
        + ufl.inner(u, v) * ufl.ds(subdomain_data=ft, subdomain_id=(1,2))


    start = time.perf_counter()
    num_batches = 100
    Fs = create_batched_form(F, num_batches=num_batches)    
    b_batched = dolfinx.fem.petsc.create_vector(V)
    for Fi in Fs:
        dolfinx.fem.petsc.assemble_vector(b_batched, Fi)
    b_batched.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    end = time.perf_counter()

    MPI.COMM_WORLD.Barrier()
    start_ref = time.perf_counter()
    F_ref = dolfinx.fem.form(F)
    b_ref = dolfinx.fem.petsc.assemble_vector(F_ref)
    b_ref.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    end_ref = time.perf_counter()

    coeffs = dolfinx.fem.pack_coefficients(F_ref)
    max_mem = 0
    for key, value in coeffs.items():
        if value.nbytes > max_mem:
            max_mem = value.nbytes

    max_mem_b = 0
    for i in range(num_batches):
        batched_coeff = dolfinx.fem.pack_coefficients(Fs[i])
        for key, value in batched_coeff.items():
            if max_mem_b < value.nbytes:
                max_mem_b = value.nbytes

    print(max_mem, max_mem_b, max_mem/max_mem_b)
    print(f"Batched {end-start}, Ref {end_ref-start_ref}")
    np.testing.assert_allclose(b_ref.array, b_batched.array)
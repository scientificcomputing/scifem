from mpi4py import MPI
import dolfinx.fem.petsc
import ufl
import numpy as np

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), np.ones(num_cells_local, dtype=np.int32))
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
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
                        entities = np.arange(num_entities_local)
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

def create_idata_batches(idata: dict[dolfinx.fem.IntegralType, list[tuple[int, np.ndarray]]], max_batches:int = 10, min_batch_size: int = 10) -> list[dict[dolfinx.fem.IntegralType, list[tuple[int, np.ndarray]]]]:
    # batched_integral_data = []
    # for itg_type, (tag, integration_entities) in idata.items():

    #     breakpoint()
    pass

# Work on exterior facets tomorrow
def create_batched_form(F: ufl.Form, num_batches: int, entity_maps:list[dolfinx.mesh.EntityMap]=None) -> list[dolfinx.fem.Form]:
    new_form, integral_data = extract_integration_domains(F)
    # batched_integral_data = create_idata_batches(integral_data)
    # breakpoint()
    function_spaces = [arg.ufl_function_space() for arg in new_form.arguments()]
    coefficient_map = {coeff:coeff for coeff in new_form.coefficients()}
    constant_map = {const: const for const in new_form.constants()}
    compiled_form = dolfinx.fem.compile_form(MPI.COMM_WORLD, new_form)
    F_batch = dolfinx.fem.create_form(compiled_form, function_spaces=function_spaces,
                                      msh=mesh, subdomains=integral_data,
                                      coefficient_map=coefficient_map, constant_map=constant_map,
                                      entity_maps=entity_maps)
    return F_batch  

F_c = dolfinx.fem.form(F)
Fs = create_batched_form(F, 10)




b = dolfinx.fem.petsc.assemble_vector(F_c)

b_batched = dolfinx.fem.petsc.create_vector(V)
# for batch in F_batched:
#     dolfinx.fem.petsc.assemble_vector(b_batched, batch)
dolfinx.fem.petsc.assemble_vector(b_batched, Fs)

np.testing.assert_allclose(b.array, b_batched.array)
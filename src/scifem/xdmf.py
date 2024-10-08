from __future__ import annotations

import typing
import xml.etree.ElementTree as ET
import contextlib
from pathlib import Path
import os
import warnings

from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import dolfinx


class XDMFData(typing.Protocol):
    @property
    def points(self) -> npt.NDArray[np.float64]: ...

    @property
    def bs(self) -> int: ...

    @property
    def num_dofs_global(self) -> int: ...


def write_xdmf(
    functions: typing.Sequence[dolfinx.fem.Function],
    filename: os.PathLike,
    h5name: Path,
    xdmfdata: XDMFData,
) -> None:
    """Write the XDMF file for the point cloud.

    Args:
        functions: List of functions to write to the point cloud.
        filename: The name of the file to write the XDMF to.
        h5name: The name of the HDF5 file.
        xdmfdata: The XDMF data.

    Note:
        This function does not check the validity of the input data, i.e.
        all functions in ``functions`` are assumed to share the same function space,
        including ``block_size``.
    """

    xdmf = ET.Element("XDMF")
    xdmf.attrib["Version"] = "3.0"
    xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
    domain = ET.SubElement(xdmf, "Domain")
    grid = ET.SubElement(domain, "Grid")
    grid.attrib["GridType"] = "Uniform"
    grid.attrib["Name"] = "Point Cloud"
    topology = ET.SubElement(grid, "Topology")
    topology.attrib["NumberOfElements"] = str(xdmfdata.num_dofs_global)
    topology.attrib["TopologyType"] = "PolyVertex"
    topology.attrib["NodesPerElement"] = "1"
    top_data = ET.SubElement(topology, "DataItem")
    top_data.attrib["Dimensions"] = f"{xdmfdata.num_dofs_global} {1}"
    top_data.attrib["Format"] = "HDF"
    top_data.text = f"{h5name.name}:/Step0/Cells"
    geometry = ET.SubElement(grid, "Geometry")
    geometry.attrib["GeometryType"] = "XYZ"
    for u in functions:
        it0 = ET.SubElement(geometry, "DataItem")
        it0.attrib["Dimensions"] = f"{xdmfdata.num_dofs_global} {xdmfdata.points.shape[1]}"
        it0.attrib["Format"] = "HDF"
        it0.text = f"{h5name.name}:/Step0/Points"
        attrib = ET.SubElement(grid, "Attribute")
        attrib.attrib["Name"] = u.name
        out_bs = xdmfdata.bs
        if out_bs == 1:
            attrib.attrib["AttributeType"] = "Scalar"
        else:
            if out_bs == 2:
                # Pad to 3D
                out_bs = 3
            attrib.attrib["AttributeType"] = "Vector"
        attrib.attrib["Center"] = "Node"
        it1 = ET.SubElement(attrib, "DataItem")
        it1.attrib["Dimensions"] = f"{xdmfdata.num_dofs_global} {out_bs}"
        it1.attrib["Format"] = "HDF"
        it1.text = f"{h5name.name}:/Step0/Values_{u.name}"
    text = [
        '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
        ET.tostring(xdmf, encoding="unicode"),
    ]
    Path(filename).with_suffix(".xdmf").write_text("".join(text))


class FunctionSpaceData(typing.NamedTuple):
    """Data class for function space information."""

    points: npt.NDArray[np.float64]
    bs: int
    num_dofs_global: int
    num_dofs_local: int
    local_range: npt.NDArray[np.int64]
    comm: MPI.Intracomm

    @property
    def points_out(self) -> npt.NDArray[np.float64]:
        """Pad points to be 3D"""
        return np.ascontiguousarray(self.points[: self.num_dofs_local, :])


@contextlib.contextmanager
def h5pyfile(h5name, filemode="r", force_serial: bool = False, comm=None):
    """Context manager for opening an HDF5 file with h5py.

    Args:
        h5name: The name of the HDF5 file.
        filemode: The file mode.
        force_serial: Force serial access to the file.
        comm: The MPI communicator

    """
    import h5py

    if comm is None:
        comm = MPI.COMM_WORLD

    if h5py.h5.get_config().mpi and comm.size > 1 and not force_serial:
        h5file = h5py.File(h5name, filemode, driver="mpio", comm=comm)
    else:
        if comm.size > 1 and not force_serial:
            warnings.warn(
                "h5py is not installed with MPI support, while using {comm.size} processes"
            )
        h5file = h5py.File(h5name, filemode)
    yield h5file
    h5file.close()


def write_hdf5_h5py(
    functions: typing.Sequence[dolfinx.fem.Function],
    h5name: Path,
    data: FunctionSpaceData | None = None,
) -> None:
    """Write the point cloud to an HDF5 file using h5py.

    Args:
        functions: The functions to write to the point cloud.
        h5name: The name of the file to write the point cloud to.
        data: The function space data.

    Note:
        All input ``functions`` has to share the same function space.
    """
    if len(functions) == 0:
        return

    if data is None:
        data = check_function_space(functions)
    if data is None:
        warnings.warn("No functions to write to point cloud")
        return

    with h5pyfile(h5name=h5name, filemode="w", comm=data.comm) as h5file:
        step = h5file.create_group(np.bytes_("Step0"))
        points = step.create_dataset(
            "Points", (data.num_dofs_global, data.points.shape[1]), dtype=data.points.dtype
        )
        points[data.local_range[0] : data.local_range[1], :] = data.points_out
        cells = step.create_dataset("Cells", (data.num_dofs_global,), dtype=np.int64)
        cells[data.local_range[0] : data.local_range[1]] = np.arange(
            data.local_range[0], data.local_range[1], dtype=np.int64
        )
        for u in functions:
            # Pad array to 3D if vector space with 2 components
            array = np.zeros(
                (data.num_dofs_local, data.bs if data.bs != 2 else 3), dtype=u.x.array.dtype
            )
            array[:, : data.bs] = u.x.array[: data.num_dofs_local * data.bs].reshape(-1, data.bs)
            dset = step.create_dataset(
                f"Values_{u.name}", (data.num_dofs_global, array.shape[1]), dtype=array.dtype
            )
            dset[data.local_range[0] : data.local_range[1], :] = array


def write_hdf5_adios(
    functions: typing.Sequence[dolfinx.fem.Function],
    h5name: Path,
    data: FunctionSpaceData | None = None,
) -> None:
    """Write the point cloud to an HDF5 file using ADIOS2.

    Args:
        functions: The functions to write to the point cloud.
        h5name: The name of the file to write the point cloud to.
        data: The function space data.

    Note:
        All input ``functions`` has to share the same function space.
    """
    import adios2

    def resolve_adios_scope(adios2):
        return adios2 if not hasattr(adios2, "bindings") else adios2.bindings

    adios2 = resolve_adios_scope(adios2)

    if len(functions) == 0:
        return

    if data is None:
        data = check_function_space(functions)
    if data is None:
        warnings.warn("No functions to write to point cloud")
        return

    # Create ADIOS2 reader
    adios = adios2.ADIOS(data.comm)
    io = adios.DeclareIO("Point cloud writer")
    io.SetEngine("HDF5")
    outfile = io.Open(h5name.as_posix(), adios2.Mode.Write)

    pointvar = io.DefineVariable(
        "Points",
        data.points_out,
        shape=[data.num_dofs_global, data.points.shape[1]],
        start=[data.local_range[0], 0],
        count=[data.num_dofs_local, data.points.shape[1]],
    )
    outfile.Put(pointvar, data.points_out)
    cells = np.arange(data.local_range[0], data.local_range[1], dtype=np.int64)
    cellvar = io.DefineVariable(
        "Cells",
        cells,
        shape=[data.num_dofs_global],
        start=[data.local_range[0]],
        count=[data.num_dofs_local],
    )
    outfile.Put(cellvar, cells)

    for u in functions:
        array = np.zeros(
            (data.num_dofs_local, data.bs if data.bs != 2 else 3), dtype=u.x.array.dtype
        )
        array[:, : data.bs] = u.x.array[: data.num_dofs_local * data.bs].reshape(-1, data.bs)

        valuevar = io.DefineVariable(
            f"Values_{u.name}",
            array,
            shape=[data.num_dofs_global, array.shape[1]],
            start=[data.local_range[0], 0],
            count=[data.num_dofs_local, array.shape[1]],
        )
        outfile.Put(valuevar, array)
    outfile.PerformPuts()
    outfile.Close()
    assert adios.RemoveIO("Point cloud writer")


def create_function_space_data(V: dolfinx.fem.FunctionSpace) -> FunctionSpaceData:
    """Create function space data from a function space.

    Args:
        V: The function space.

    Returns:
        The function space data.

    """
    points = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs
    index_map = V.dofmap.index_map
    comm = V.mesh.comm

    return FunctionSpaceData(
        points=points,
        bs=bs,
        num_dofs_global=index_map.size_global,
        num_dofs_local=index_map.size_local,
        local_range=np.array(index_map.local_range, dtype=np.int64),
        comm=comm,
    )


def check_function_space(
    functions: typing.Sequence[dolfinx.fem.Function],
) -> FunctionSpaceData | None:
    """Check that all functions are in the same function space,
    and return the function space data.

    Args:
        functions: The functions to check.

    Returns:
        The function space data if all functions are in the same function space, otherwise None.

    """
    if len(functions) == 0:
        return None

    # Check that all functions are in the same function space
    u = functions[0]
    for v in functions[1:]:
        if u.function_space != v.function_space:
            raise ValueError("All functions must be in the same function space.")

    return create_function_space_data(u.function_space)


def create_pointcloud(
    filename: os.PathLike, functions: typing.Sequence[dolfinx.fem.Function]
) -> None:
    """Create a point cloud from a list of functions to be visualized in Paraview.
    The point cloud is written to a file in XDMF format.

    Args:
        filename: The name of the file to write the point cloud to.
        functions: The functions to write to the point cloud.

    Note:
        This is useful for visualizing functions in quadrature spaces.

    Note:
        Any function space that can call `tabulate_dof_coordinates` can be used.

    Note:
        ADIOS2 is the preferred backend for writing HDF5 files, and will be used if available.
        If ADIOS2 is not available, `h5py` will be used.
    """
    data = check_function_space(functions)
    if data is None:
        warnings.warn("No functions to write to point cloud")
        return

    h5name = Path(filename).with_suffix(".h5")

    # Write XDMF on rank 0
    if data.comm.rank == 0:
        write_xdmf(functions, filename, h5name, data)

    try:
        write_hdf5_adios(functions=functions, h5name=h5name, data=data)
    except (ImportError, TypeError, ValueError):
        warnings.warn("ADIOS2 not available, using h5py")
        write_hdf5_h5py(functions=functions, h5name=h5name, data=data)

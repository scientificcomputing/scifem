from __future__ import annotations

from collections import defaultdict
import logging
import typing
import xml.etree.ElementTree as ET
import contextlib
from pathlib import Path
import os
import abc
import functools
import warnings

from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import dolfinx
import basix
import ufl

logger = logging.getLogger(__name__)


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        msg = (
            "Call to deprecated function {}.".format(func.__name__),
            "Please use the class scifem.xdmf.XDMFFile instead.",
        )
        logger.warning(msg)
        warnings.warn(
            msg,
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


class XDMFData(typing.Protocol):
    @property
    def points(self) -> npt.NDArray[np.float64]: ...

    @property
    def bs(self) -> int: ...

    @property
    def num_dofs_global(self) -> int: ...


@deprecated
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
            raise ValueError(
                f"h5py is not installed with MPI support, while using {comm.size} processes.",
                "If you really want to do this, turn on the `force_serial` flag.",
            )
        h5file = h5py.File(h5name, filemode)
    yield h5file
    h5file.close()


@deprecated
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


@deprecated
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
    gdim = V.mesh.geometry.dim

    return FunctionSpaceData(
        points=points[:, :gdim],
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


@deprecated
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
        warnings.warn("ADIOS2 not available, using h5py", stacklevel=2)
        write_hdf5_h5py(functions=functions, h5name=h5name, data=data)


def read_time_values_from_xdmf(xdmfname: os.PathLike) -> dict[float, int]:
    """Read time values from an XDMF file.

    Args:
        xdmfname: The name of the XDMF file.

    Returns:
        A dictionary with time values as keys and step numbers as values.

    """
    tree = ET.parse(xdmfname)
    current_name = ""
    all_times_values: dict[str, dict[float, int]] = defaultdict(dict)
    index = 0
    for elem in tree.iter():
        if elem.tag == "Attribute":
            new_current_name = elem.get("Name", "")
            if new_current_name != current_name:
                current_name = new_current_name
                index = 0
        if elem.tag == "Time":
            time_value = float(elem.get("Value", 0.0))
            if current_name == "":
                raise ValueError(f"No name found for time value {time_value}")

            all_times_values[current_name][time_value] = index
            index += 1

    # Check if all time values are the same
    first_time_values = list(all_times_values.values())[0]
    for name, time_values in all_times_values.items():
        if time_values != first_time_values:
            raise ValueError(f"Time values for {name} are not the same")

    return first_time_values


# Taken from adios4dolfinx
adios_to_numpy_dtype = {
    "float": np.float32,
    "double": np.float64,
    "float complex": np.complex64,
    "double complex": np.complex128,
    "uint32_t": np.uint32,
}


def create_point_mesh(
    comm: MPI.Intracomm,
    points: npt.NDArray[np.floating],
    cells: npt.NDArray[np.floating] | None = None,
) -> dolfinx.mesh.Mesh:
    """
    Create a mesh consisting of points only.

    Note:
        No nodes are shared between processes.

    Args:
        comm: MPI communicator to create the mesh on.
        points: Points local to the process in the mesh.
    """
    # Create mesh topology
    if cells is None:
        cells = np.arange(points.shape[0], dtype=np.int32).reshape(-1, 1)
    topology = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, dolfinx.mesh.CellType.point)
    num_nodes_local = cells.shape[0]
    imap = dolfinx.common.IndexMap(MPI.COMM_WORLD, num_nodes_local)
    local_range = imap.local_range[0]
    igi = np.arange(num_nodes_local, dtype=np.int64) + local_range
    topology.set_index_map(0, imap)
    topology.set_connectivity(dolfinx.graph.adjacencylist(cells.astype(np.int32)), 0, 0)

    # Create mesh geometry
    e = basix.ufl.element("Lagrange", "point", 0, shape=(points.shape[1],))
    c_el = dolfinx.fem.coordinate_element(e.basix_element)
    geometry = dolfinx.mesh.create_geometry(imap, cells, c_el._cpp_object, points, igi)

    # Create DOLFINx mesh
    if points.dtype == np.float64:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(comm, topology, geometry._cpp_object)
    elif points.dtype == np.float32:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float32(comm, topology, geometry._cpp_object)
    else:
        raise RuntimeError(f"Unsupported dtype for mesh {points.dtype}")
    # Wrap as Python object
    return dolfinx.mesh.Mesh(cpp_mesh, domain=ufl.Mesh(e))


def compute_local_range(comm: MPI.Intracomm, N: np.int64):
    """
    Divide a set of `N` objects into `M` partitions, where `M` is
    the size of the MPI communicator `comm`.

    NOTE: If N is not divisible by the number of ranks, the first `r`
    processes gets an extra value

    Returns the local range of values
    """
    rank = comm.rank
    size = comm.size
    n = N // size
    r = N % size
    # First r processes has one extra value
    if rank < r:
        return [rank * (n + 1), (rank + 1) * (n + 1)]
    else:
        return [rank * n + r, (rank + 1) * n + r]


class BaseXDMFFile(abc.ABC):
    filename: os.PathLike
    filemode: typing.Literal["r", "a", "w"]
    backend: typing.Literal["h5py", "adios2"]
    _data: FunctionSpaceData
    _time_values: dict[float, int]

    @property
    @abc.abstractmethod
    def data_names(self) -> list[str]:
        """The names of the data."""

    @property
    @abc.abstractmethod
    def data_arrays(self) -> list[npt.NDArray[np.floating]]:
        """The data arrays."""
        ...

    def __post_init__(self) -> None:
        self.h5name = Path(self.filename).with_suffix(".h5")
        self.xdmfname = Path(self.filename).with_suffix(".xdmf")

        if self.filemode == "r":
            if not self.h5name.exists():
                raise FileNotFoundError(f"{self.h5name} does not exist")
            if not self.xdmfname.exists():
                raise FileNotFoundError(f"{self.xdmfname} does not exist")

            # Read time values from XDMF file

        elif self.filemode == "w":
            # Overwrite existing files so make sure they don't exist
            self.h5name.unlink(missing_ok=True)
            self.xdmfname.unlink(missing_ok=True)
        else:
            raise NotImplementedError(f"Filemode {self.filemode} not supported")

    def _init_backend(self) -> None:
        assert self.backend in [
            "h5py",
            "adios2",
        ], f"Unknown backend {self.backend}, must be 'h5py' or 'adios2'"
        if self.backend == "adios2":
            try:
                self._init_adios()
            except (ImportError, TypeError, ValueError):
                msg = "ADIOS2 not available, using h5py"
                warnings.warn(msg)
                logging.warning(msg)
                self.backend = "h5py"
        if self.backend == "h5py":
            self._init_h5py()

        self._init_time_values()

    def _init_time_values(self) -> None:
        if self.filemode == "r":
            self._time_values = read_time_values_from_xdmf(self.xdmfname)
        elif self.filemode == "w":
            self._time_values = {}
        else:
            raise NotImplementedError(f"Filemode {self.filemode} not supported")

    def _init_h5py(self) -> None:
        logger.debug("Initializing h5py")
        self._outfile = h5pyfile(
            h5name=self.h5name, filemode=self.filemode, comm=self._data.comm
        ).__enter__()
        if self.filemode == "r":
            assert "Step0" in self._outfile, "Step0 not found in HDF5 file"
            self._step = self._outfile["Step0"]
            return None

        self._step = self._outfile.create_group(np.bytes_("Step0"))
        points = self._step.create_dataset(
            "Points",
            (self._data.num_dofs_global, self._data.points.shape[1]),
            dtype=self._data.points.dtype,
        )
        points[self._data.local_range[0] : self._data.local_range[1], :] = self._data.points_out
        cells = self._step.create_dataset("Cells", (self._data.num_dofs_global,), dtype=np.int64)
        cells[self._data.local_range[0] : self._data.local_range[1]] = np.arange(
            self._data.local_range[0], self._data.local_range[1], dtype=np.int64
        )

    def _write_h5py(self, index: int) -> None:
        logger.debug(f"Writing h5py at time {index}")
        for data_name, data_array in zip(self.data_names, self.data_arrays):
            # Pad array to 3D if vector space with 2 components
            array = np.zeros(
                (self._data.num_dofs_local, self._data.bs if self._data.bs != 2 else 3),
                dtype=data_array.dtype,
            )
            array[:, : self._data.bs] = data_array[
                : self._data.num_dofs_local * self._data.bs
            ].reshape(-1, self._data.bs)
            dset = self._step.create_dataset(
                f"Values_{data_name}_{index}",
                (self._data.num_dofs_global, array.shape[1]),
                dtype=array.dtype,
            )
            dset[self._data.local_range[0] : self._data.local_range[1], :] = array

    def _read_h5py(self, index: int) -> None:
        logger.debug(f"Writing h5py at time {index}")

        cells = self._step["Cells"]
        points = self._step["Points"]
        assert cells.shape[0] == points.shape[0]
        local_range = compute_local_range(self._data.comm, cells.shape[0])
        cells_local = cells[local_range[0] : local_range[1]]
        points_local = points[local_range[0] : local_range[1], :]
        point_mesh = create_point_mesh(
            comm=self._data.comm,
            points=points_local,
            cells=cells_local.reshape(-1, 1),
        )

        if self._data.bs == 1:
            shape: tuple[int, ...] = ()
        else:
            shape = (self._data.bs,)

        V = dolfinx.fem.functionspace(point_mesh, ("DG", 0, shape))
        self.vs = []
        for data_name in self.data_names:
            v = dolfinx.fem.Function(V, name=data_name)
            # Pad array to 3D if vector space with 2 components
            key = f"Values_{data_name}_{index}"
            if key not in self._step:
                raise ValueError(f"Variable {data_name} not found in HDF5 file")

            v.x.array[:] = self._step[f"Values_{data_name}_{index}"][
                local_range[0] : local_range[1], : self._data.bs
            ].flatten()
            self.vs.append(v)

    def _close_h5py(self) -> None:
        logger.debug("Closing HDF5 file")
        self._outfile.close()

    def _open_adios(self):
        self._adios = self.adios2.ADIOS(self._data.comm)
        self._io = self._adios.DeclareIO("Point cloud writer")
        self._io.SetEngine("HDF5")
        mode = self.adios2.Mode.Write if self.filemode == "w" else self.adios2.Mode.Read
        self._outfile = self._io.Open(self.h5name.as_posix(), mode)

    def _init_adios(self) -> None:
        logger.debug("Initializing ADIOS2")
        import adios2

        def resolve_adios_scope(adios2):
            return adios2 if not hasattr(adios2, "bindings") else adios2.bindings

        self.adios2 = resolve_adios_scope(adios2)

        # Create ADIOS2 reader
        if self.filemode == "r":
            return None
        self._open_adios()
        pointvar = self._io.DefineVariable(
            "Points",
            self._data.points_out,
            shape=[self._data.num_dofs_global, self._data.points.shape[1]],
            start=[self._data.local_range[0], 0],
            count=[self._data.num_dofs_local, self._data.points.shape[1]],
        )
        self._outfile.Put(pointvar, self._data.points_out)
        cells = np.arange(self._data.local_range[0], self._data.local_range[1], dtype=np.int64)
        cellvar = self._io.DefineVariable(
            "Cells",
            cells,
            shape=[self._data.num_dofs_global],
            start=[self._data.local_range[0]],
            count=[self._data.num_dofs_local],
        )
        self._outfile.Put(cellvar, cells)

    def _write_adios(self, index: int) -> None:
        logger.debug(f"Writing adios at time {index}")
        for data_name, data_array in zip(self.data_names, self.data_arrays):
            array = np.zeros(
                (self._data.num_dofs_local, self._data.bs if self._data.bs != 2 else 3),
                dtype=data_array.dtype,
            )
            array[:, : self._data.bs] = data_array[
                : self._data.num_dofs_local * self._data.bs
            ].reshape(-1, self._data.bs)

            valuevar = self._io.DefineVariable(
                f"Values_{data_name}_{index}",
                array,
                shape=[self._data.num_dofs_global, array.shape[1]],
                start=[self._data.local_range[0], 0],
                count=[self._data.num_dofs_local, array.shape[1]],
            )
            self._outfile.Put(valuevar, array)
        self._outfile.PerformPuts()

    def _read_adios(self, index: int) -> None:
        logger.debug(f"Reading adios at time {index}")
        self._open_adios()
        hit = False
        for data_name, data_array in zip(self.data_names, self.data_arrays):
            variable_name = f"Values_{data_name}_{index}"
            for i in range(self._outfile.Steps()):
                self._outfile.BeginStep()
                if variable_name in self._io.AvailableVariables().keys():
                    arr = self._io.InquireVariable(variable_name)
                    arr_shape = arr.Shape()
                    vals = np.empty(arr_shape, dtype=adios_to_numpy_dtype[arr.Type()])

                    self._outfile.Get(arr, vals, self.adios2.Mode.Sync)
                    start = self._data.local_range[0]
                    end = self._data.local_range[0] + self._data.num_dofs_local
                    data_array[: self._data.num_dofs_local * self._data.bs] = vals[
                        start:end, : self._data.bs
                    ].flatten()
                    hit = True
                    self._outfile.EndStep()
                    break
                else:
                    self._outfile.EndStep()
            else:
                self._outfile.EndStep()
                break

        self._close_adios()
        if not hit:
            raise ValueError(f"Variable {variable_name} not found in ADIOS2 file")

    def _close_adios(self) -> None:
        logger.debug("Closing ADIOS2 file")
        try:
            self._outfile.Close()
            assert self._adios.RemoveIO("Point cloud writer")
        except ValueError:
            # File is allready closed
            logger.debug("ADIOS2 file already closed")

    def close(self) -> None:
        """Close the XDMF file."""
        logger.debug("Closing XDMF file")
        if self.backend == "adios2":
            self._close_adios()
        elif self.backend == "h5py":
            self._close_h5py()

    def _write_xdmf(self) -> None:
        logger.debug("Writing XDMF file")
        xdmf = ET.Element("Xdmf")
        xdmf.attrib["Version"] = "3.0"
        xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
        domain = ET.SubElement(xdmf, "Domain")
        grid = ET.SubElement(domain, "Grid")
        grid.attrib["GridType"] = "Uniform"
        grid.attrib["Name"] = "Point Cloud"
        topology = ET.SubElement(grid, "Topology")
        topology.attrib["NumberOfElements"] = str(self._data.num_dofs_global)
        topology.attrib["TopologyType"] = "PolyVertex"
        topology.attrib["NodesPerElement"] = "1"
        top_data = ET.SubElement(topology, "DataItem")
        top_data.attrib["Dimensions"] = f"{self._data.num_dofs_global} {1}"
        top_data.attrib["Format"] = "HDF"
        top_data.text = f"{self.h5name.name}:/Step0/Cells"
        geometry = ET.SubElement(grid, "Geometry")
        if self._data.points.shape[1] == 2:
            geometry.attrib["GeometryType"] = "XY"
        elif self._data.points.shape[1] == 3:
            geometry.attrib["GeometryType"] = "XYZ"
        else:
            raise ValueError(f"Unsupported geometry type {self._data.points.shape[1]}")
        it0 = ET.SubElement(geometry, "DataItem")
        it0.attrib["Dimensions"] = f"{self._data.num_dofs_global} {self._data.points.shape[1]}"
        it0.attrib["Format"] = "HDF"
        it0.text = f"{self.h5name.name}:/Step0/Points"
        for name in self.data_names:
            ugrid_collection = ET.SubElement(domain, "Grid")
            ugrid_collection.attrib["GridType"] = "Collection"
            ugrid_collection.attrib["CollectionType"] = "Temporal"
            ugrid_collection.attrib["Name"] = name

            for step, time_value in enumerate(sorted(self._time_values)):
                ugrid = ET.SubElement(ugrid_collection, "Grid")
                ugrid.attrib["GridType"] = "Uniform"
                ugrid.attrib["Name"] = name
                xp = ET.SubElement(ugrid, "xi:include")
                xp.attrib["xpointer"] = (
                    "xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology or self::Geometry])"  # noqa: E501
                )
                attrib = ET.SubElement(ugrid, "Attribute")
                attrib.attrib["Name"] = name
                time = ET.SubElement(ugrid, "Time")
                time.attrib["Value"] = str(time_value)
                out_bs = self._data.bs
                if out_bs == 1:
                    attrib.attrib["AttributeType"] = "Scalar"
                else:
                    if out_bs == 2:
                        # Pad to 3D
                        out_bs = 3
                    attrib.attrib["AttributeType"] = "Vector"
                attrib.attrib["Center"] = "Node"
                it1 = ET.SubElement(attrib, "DataItem")
                it1.attrib["Dimensions"] = f"{self._data.num_dofs_global} {out_bs}"
                it1.attrib["Format"] = "HDF"
                it1.text = f"{self.h5name.name}:/Step0/Values_{name}_{step}"

        ET.indent(xdmf)
        text = [
            '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
            ET.tostring(xdmf, encoding="unicode"),
        ]
        self.xdmfname.write_text("".join(text))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def write(self, time: float) -> None:
        """Write the point cloud at a given time.

        Args:
            time: The time value.

        """
        logger.debug(f"Writing time {time}")
        time = float(time)
        if time in self._time_values:
            msg = f"Time {time} already written to file. Skipping."
            logger.warning(msg)
            return

        index = len(self._time_values)
        self._time_values[time] = index

        self._write_xdmf()
        if self.backend == "adios2":
            self._write_adios(index)
        elif self.backend == "h5py":
            self._write_h5py(index)

    def read(self, time: float) -> None:
        """Read the point cloud at a given time.

        Args:
            time: The time value.

        """
        logger.debug(f"Writing time {time}")
        time = float(time)
        if time not in self._time_values:
            msg = f"Time {time} not found in file."
            logger.warning(msg)
            return

        index = self._time_values[time]

        if self.backend == "adios2":
            self._read_adios(index)
        elif self.backend == "h5py":
            self._read_h5py(index)


class XDMFFile(BaseXDMFFile):
    def __init__(
        self,
        filename: os.PathLike,
        functions: typing.Sequence[dolfinx.fem.Function],
        filemode: typing.Literal["r", "a", "w"] = "w",
        backend: typing.Literal["h5py", "adios2"] = "adios2",
    ) -> None:
        self.functions = functions
        self.filename = filename
        self.filemode = filemode
        self.backend = backend
        super().__post_init__()
        if len(self.functions) == 0:
            raise ValueError("No functions to write to point cloud")
        for f in self.functions:
            if not isinstance(f, dolfinx.fem.Function):
                raise ValueError("All functions must be of type dolfinx.fem.Function")

        data = check_function_space(self.functions)
        if data is None:
            raise ValueError("All functions must be in the same function space")
        self._data = data

        self._init_backend()

    @property
    def data_names(self) -> list[str]:
        return [f.name for f in self.functions]

    @property
    def data_arrays(self) -> list[npt.NDArray[np.floating]]:
        return [f.x.array for f in self.functions]

    def _read_h5py(self, index: int) -> None:
        super()._read_h5py(index)
        for v, f in zip(self.vs, self.functions):
            cell_map = f.function_space.mesh.topology.index_map(f.function_space.mesh.topology.dim)
            num_cells = cell_map.size_local + cell_map.num_ghosts
            cells = np.arange(num_cells, dtype=np.int32)
            data = dolfinx.fem.create_interpolation_data(
                f.function_space, v.function_space, cells, padding=1e-10
            )

            f.interpolate_nonmatching(v, cells, data)


class NumpyXDMFFile(BaseXDMFFile):
    def __init__(
        self,
        filename: os.PathLike,
        arrays: list[npt.NDArray[np.floating]],
        function_space_data: FunctionSpaceData,
        filemode: typing.Literal["r", "a", "w"] = "w",
        backend: typing.Literal["h5py", "adios2"] = "adios2",
        array_names: list[str] | None = None,
    ) -> None:
        if len(arrays) == 0:
            raise ValueError("No arrays to write to point cloud")
        for f in arrays:
            if not isinstance(f, np.ndarray):
                raise ValueError("All arrays must be of type numpy.ndarray")
        f0 = arrays[0]
        for f in arrays[1:]:
            if f.shape != f0.shape:
                raise ValueError("All arrays must have the same shape")

        self._data_arrays = arrays
        self._data = function_space_data
        # FIXME: Should we do some checks here on the data?

        if array_names is None:
            array_names = [f"array_{i}" for i in range(len(arrays))]
        self._data_names = array_names

        self.filename = filename
        self.filemode = filemode
        self.backend = backend
        super().__post_init__()

        self._init_backend()

    @property
    def data_names(self) -> list[str]:
        return self._data_names

    @property
    def data_arrays(self) -> list[npt.NDArray[np.floating]]:
        return self._data_arrays

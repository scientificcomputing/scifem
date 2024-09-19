import typing
import xml.etree.ElementTree as ET
from pathlib import Path
import os

import numpy as np
import numpy.typing as npt
import dolfinx


def write_xdmf(
    us: typing.Sequence[dolfinx.fem.Function],
    filename: os.PathLike,
    num_dofs_global: int,
    points: npt.NDArray[np.float64],
    bs: int,
    h5name: Path,
) -> None:
    xdmf = ET.Element("XDMF")
    xdmf.attrib["Version"] = "3.0"
    xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
    domain = ET.SubElement(xdmf, "Domain")
    grid = ET.SubElement(domain, "Grid")
    grid.attrib["GridType"] = "Uniform"
    grid.attrib["Name"] = "Point Cloud"
    topology = ET.SubElement(grid, "Topology")
    topology.attrib["NumberOfElements"] = str(num_dofs_global)
    topology.attrib["TopologyType"] = "PolyVertex"
    topology.attrib["NodesPerElement"] = "1"
    geometry = ET.SubElement(grid, "Geometry")
    geometry.attrib["GeometryType"] = "XY" if points.shape[1] == 2 else "XYZ"
    for u in us:
        it0 = ET.SubElement(geometry, "DataItem")
        it0.attrib["Dimensions"] = f"{num_dofs_global} {points.shape[1]}"
        it0.attrib["Format"] = "HDF"
        it0.text = f"{h5name.name}:/Step0/Points"
        attrib = ET.SubElement(grid, "Attribute")
        attrib.attrib["Name"] = u.name
        if bs == 1:
            attrib.attrib["AttributeType"] = "Scalar"
        else:
            attrib.attrib["AttributeType"] = "Vector"
        attrib.attrib["Center"] = "Node"
        it1 = ET.SubElement(attrib, "DataItem")
        it1.attrib["Dimensions"] = f"{num_dofs_global} {bs}"
        it1.attrib["Format"] = "HDF"
        it1.text = f"{h5name.name}:/Step0/Values_{u.name}"
    text = [
        '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
        ET.tostring(xdmf, encoding="unicode"),
    ]
    Path(filename).with_suffix(".xdmf").write_text("".join(text))


def create_pointcloud(filename: os.PathLike, us: typing.Sequence[dolfinx.fem.Function]) -> None:
    """Create a point cloud from a list of functions to be visualized in Paraview.
    The point cloud is written to a file in XDMF format.

    Args:
        filename: The name of the file to write the point cloud to.
        us: The functions to write to the point cloud.

    Note:
        This is useful for visualizing functions in quadrature spaces.

    """
    # Adopted from https://gist.github.com/jorgensd/8bae61ad7a0c211570dff0116a68a356
    if len(us) == 0:
        return

    u = us[0]
    points = u.function_space.tabulate_dof_coordinates()
    h5name = Path(filename).with_suffix(".h5")

    bs = u.function_space.dofmap.index_map_bs
    comm = u.function_space.mesh.comm
    num_dofs_global = u.function_space.dofmap.index_map.size_global
    num_dofs_local = u.function_space.dofmap.index_map.size_local
    local_range = np.array(u.function_space.dofmap.index_map.local_range, dtype=np.int64)

    # Write XDMF on rank 0
    if comm.rank == 0:
        write_xdmf(us, filename, num_dofs_global, points, bs, h5name)

    import adios2

    def resolve_adios_scope(adios2):
        return adios2 if not hasattr(adios2, "bindings") else adios2.bindings

    adios2 = resolve_adios_scope(adios2)

    # Create ADIOS2 reader
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("Point cloud writer")
    io.SetEngine("HDF5")
    outfile = io.Open(h5name.as_posix(), adios2.Mode.Write)
    points_out = points[:num_dofs_local, :]

    pointvar = io.DefineVariable(
        "Points",
        points_out,
        shape=[num_dofs_global, points.shape[1]],
        start=[local_range[0], 0],
        count=[num_dofs_local, points.shape[1]],
    )
    outfile.Put(pointvar, points_out)
    for u in us:
        data = u.x.array[: num_dofs_local * bs].reshape(-1, bs)

        valuevar = io.DefineVariable(
            f"Values_{u.name}",
            data,
            shape=[num_dofs_global, bs],
            start=[local_range[0], 0],
            count=[num_dofs_local, bs],
        )
        outfile.Put(valuevar, data)
    outfile.PerformPuts()
    outfile.Close()
    assert adios.RemoveIO("Point cloud writer")

import gmsh
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Create a mesh for a box in a channel")
parser.add_argument(
    "--res", type=float, default=0.05, help="Resolution of the mesh (at inlet)"
)
parser.add_argument(
    "--periodic", action="store_true", help="Create periodic boundary conditions"
)
parser.add_argument("--L", type=float, default=1, help="Length of the channel")
parser.add_argument("--H", type=float, default=0.2, help="Height of the channel")
parser.add_argument(
    "--box_pos", type=float, nargs=2, default=[0.2, 0], help="Position of the box"
)
parser.add_argument(
    "--box_size", type=float, nargs=2, default=[0.15, 0.05], help="Size of the box"
)
parser.add_argument("--algorithm", type=int, default=5, help="Meshing algorithm to use")
parser.add_argument("--optimize", action="store_true", help="Optimize the mesh")
parser.add_argument("--visualize", action="store_true", help="Visualize the mesh")
parser.add_argument("--output", type=Path, default="mesh.msh", help="Output file")
parser.add_argument("--wall_marker", type=int, default=1, help="Marker for the walls")
parser.add_argument("--inlet_marker", type=int, default=2, help="Marker for the inlet")
parser.add_argument(
    "--outlet_marker", type=int, default=3, help="Marker for the outlet"
)
parser.add_argument("--quadrilateral", action="store_true", help="Use quadrilaterals")
if __name__ == "__main__":
    args = parser.parse_args()

    gmsh.initialize()

    # Create channel with box cut out
    fluid = gmsh.model.occ.add_rectangle(0, 0, 0, args.L, args.H)
    box = gmsh.model.occ.add_rectangle(*args.box_pos, 0, *args.box_size)
    gmsh.model.occ.synchronize()

    new_fluid, _ = gmsh.model.occ.cut([(2, fluid)], [(2, box)])
    gmsh.model.occ.synchronize()

    # Set various meh resolutions (finer at box than inlet)
    tol = 1e-12
    inlet_nodes = gmsh.model.getEntitiesInBoundingBox(
        0 - tol, 0, 0, tol, 1 + tol, tol, dim=0
    )
    
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
    box_nodes = gmsh.model.getEntitiesInBoundingBox(
        args.box_pos[0] - tol,
        args.box_pos[1] - tol,
        0,
        args.box_pos[0] + args.box_size[0] + tol,
        args.box_pos[1] + args.box_size[1] + tol,
        tol,
        dim=0,
    )


    # Mark each boundary
    bndry = gmsh.model.getBoundary(new_fluid, oriented=False)
    walls = []
    inlet_periodic = []
    outlet_periodic = []
    for surface in bndry:
        com = gmsh.model.occ.getCenterOfMass(*surface)
        if np.isclose(com[0], 0):
            inlet_periodic.append(surface[1])
        elif np.isclose(com[0], args.L):
            outlet_periodic.append(surface[1])
        else:
            walls.append(surface[1])

    if args.periodic:
        # Using translation matrix as specified in:
        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        translation = [1, 0, 0, args.L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, outlet_periodic, inlet_periodic, translation)
        gmsh.model.occ.synchronize()

    wall_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(wall_dist, "EdgesList", walls)
    wall_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(wall_threshold, "IField", wall_dist)
    gmsh.model.mesh.field.setNumber(wall_threshold, "LcMin", args.res)
    gmsh.model.mesh.field.setNumber(wall_threshold, "LcMax", 2*args.res)
    gmsh.model.mesh.field.setNumber(wall_threshold, "DistMin", 0.1*args.box_size[1])
    gmsh.model.mesh.field.setNumber(wall_threshold, "DistMax", args.box_size[1])
    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [wall_threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    fluid_ = [surface[1] for surface in surfaces]
    gmsh.model.addPhysicalGroup(1, walls, args.wall_marker)
    gmsh.model.addPhysicalGroup(1, inlet_periodic, args.inlet_marker)
    gmsh.model.addPhysicalGroup(1, outlet_periodic, args.outlet_marker)
    gmsh.model.addPhysicalGroup(2, fluid_, 1)
    if args.quadrilateral:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", args.algorithm)

    # We combine these fields by using the minimum field


    gmsh.model.mesh.generate(2)
    if args.optimize:
        gmsh.model.mesh.optimize("Netgen")

    if args.visualize:
        gmsh.fltk.run()

    gmsh.write(args.output.absolute().as_posix())
    gmsh.finalize()

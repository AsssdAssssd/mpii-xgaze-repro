import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

BASE = os.path.dirname(os.path.abspath(__file__))
MESH_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "..", "tmp", "meshtxt", "p00", "day01", "0425.txt")

data = np.loadtxt(MESH_FILE)[:, :3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
pcd.paint_uniform_color([1.0, 0.2, 0.2])
o3d.io.write_point_cloud("output.pcd", pcd)

mean = data.mean(axis=0)
_, _, vh = np.linalg.svd(data - mean, full_matrices=False)
pts2d = (data - mean) @ vh[:2].T
tri = Delaunay(pts2d)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(data)
mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.7, 0.7, 0.7])
o3d.io.write_triangle_mesh("output_hull.ply", mesh)

lines = []
for a, b, c in tri.simplices:
    lines.extend([[a, b], [b, c], [c, a]])
ls = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(data),
    lines=o3d.utility.Vector2iVector(np.array(lines)),
)
ls.paint_uniform_color([0.1, 0.1, 0.1])
o3d.io.write_line_set("output_hull_wire.ply", ls)

extent = np.linalg.norm(data.max(axis=0) - data.min(axis=0))
cam_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=extent * 0.5, origin=[0, 0, 0])
face_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=extent * 0.3, origin=data.mean(axis=0))

o3d.visualization.draw_geometries(
    [pcd, mesh, cam_axes, face_axes],
    mesh_show_wireframe=True, mesh_show_back_face=True,
)
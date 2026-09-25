"""Visualize a rotated face mesh.

    python preprocessing/viz/visualize_head_pose.py <rotated_mesh.txt>

<rotated_mesh.txt>  from rotated_mesh_vis/<key>.txt (exported by the pipeline)

The mesh is shifted to the face centroid so the head pose is clearly visible;
the camera coordinate frame is drawn at the origin.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial import Delaunay


def axis_lines(ax, origin, scale, width):
    for axis, color in ((0, "r"), (1, "g"), (2, "b")):
        v = np.zeros(3)
        v[axis] = scale
        seg = np.array([origin, np.asarray(origin) + v])
        ax.add_collection3d(Line3DCollection([seg], colors=color, linewidths=width))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", help="rotated mesh txt (rotated_mesh_vis/<key>.txt)")
    args = p.parse_args()

    data = np.loadtxt(args.mesh)[:, :3]
    data = data - data.mean(axis=0)          # face-centred, keeps the true 3D shape

    mean = data.mean(axis=0)
    _, _, vh = np.linalg.svd(data - mean, full_matrices=False)
    tris = Delaunay((data - mean) @ vh[:2].T).simplices

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.add_collection3d(Poly3DCollection(
        data[tris], facecolor="lightgray", edgecolor="black",
        linewidths=0.3, alpha=0.7))

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="red", s=14, depthshade=False)

    span = np.linalg.norm(data.max(0) - data.min(0))
    axis_lines(ax, np.zeros(3), span * 0.7, width=2)

    lim = span * 0.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(args.mesh)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

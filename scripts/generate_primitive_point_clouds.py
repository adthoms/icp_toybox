import os
import sys
import argparse
import open3d
import numpy as np
from typing import Optional, Tuple, Union


def getTransformFromRPYXY(roll, pitch, yaw, x, y, z) -> np.ndarray:
    R = open3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def crop_ends(pcd, z_min=-4.9, z_max=4.9) -> open3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    pcd.points = open3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd.colors = open3d.utility.Vector3dVector(colors[mask])
    return pcd


class PrimitivePointCloudGenerator:
    def __init__(self, args):
        # Init parameters
        self.T_SOURCE_TARGET = getTransformFromRPYXY(*args.T_SOURCE_TARGET)
        self.radius = args.radius
        self.height = args.height
        self.num_points = args.num_points
        self.visualize = args.visualize
        self.save = args.save

        # Init variables
        self.cap_offset = 0.1
        self.source = None
        self.target = None
        self.save_dir = "/data/primitive/"

    def generate_cylindrical_mesh(self) -> open3d.geometry.TriangleMesh:
        return open3d.geometry.TriangleMesh.create_cylinder(
            radius=self.radius, height=(self.height + self.cap_offset), resolution=100
        )

    def sample_and_crop(
        self, mesh
    ) -> Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud]:
        source = mesh.sample_points_uniformly(number_of_points=self.num_points)
        target = mesh.sample_points_uniformly(number_of_points=self.num_points)
        source = crop_ends(source, z_min=-self.height / 2, z_max=self.height / 2)
        target = crop_ends(target, z_min=-self.height / 2, z_max=self.height / 2)
        source.paint_uniform_color([1, 0, 0])  # Red
        target.paint_uniform_color([0, 0, 1])  # Blue
        return source, target

    def transform_target(self, target) -> open3d.geometry.PointCloud:
        target.transform(self.T_SOURCE_TARGET)
        return target

    def save_point_clouds(self, source, target, file_path) -> None:
        print(f"Saving point clouds to: {file_path}")
        source_path = os.path.join(file_path, "source.pcd")
        target_path = os.path.join(file_path, "target.pcd")
        open3d.io.write_point_cloud(source_path, source)
        open3d.io.write_point_cloud(target_path, target)

    def visualize_source_target(self, source, target) -> None:
        vis = open3d.visualization.Visualizer()
        vis.create_window("Source [red] and Target [blue] Clouds", 1024, 768)
        frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0, origin=[0, 0, 0]
        )
        vis.add_geometry(self.source)
        vis.add_geometry(self.target)
        vis.add_geometry(frame)
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_front([1, 1, 1])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(1.0)
        vis.run()
        vis.destroy_window()

    def run(self) -> None:
        mesh = self.generate_cylindrical_mesh()
        self.source, self.target = self.sample_and_crop(mesh)
        self.target = self.transform_target(self.target)
        file_path = os.path.dirname(os.getcwd()) + self.save_dir
        if self.visualize:
            self.visualize_source_target(self.source, self.target)
        if self.save:
            self.save_point_clouds(self.source, self.target, file_path)


def main(args):
    parser = argparse.ArgumentParser(
        description="Generate two point clouds (i.e., source and target), sampled from an open cylinder primitive, that are offset by a transformation T_SOURCE_TARGET specified by the user."
    )
    parser.add_argument(
        "--T_SOURCE_TARGET",
        type=float,
        nargs=6,
        metavar=("ROLL", "PITCH", "YAW", "X", "Y", "Z"),
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        help="Transformation T_SOURCE_TARGET as a list: roll pitch yaw x y z.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius of the cylinder to generate point clouds.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=10.0,
        help="Height of the cylinder to generate point clouds.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000,
        help="Number of points to sample from the cylinder.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated source and target point clouds.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the generated source and target point clouds to disk.",
    )

    args = parser.parse_args(args)
    generator = PrimitivePointCloudGenerator(args)
    generator.run()


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
import os
import open3d as o3d
import argparse
from scripts.contract_box_sampling import sample_box_points


def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.0, 0.0])
    return spheres

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fps", type=int, default=-1, help="Optional number of points to keep after farthest point sampling")
    parser.add_argument("--num_contract_split", type=int, default=-1, help="Optional number of height splits to make for keeping keypoints near bounding boxes")
    parser.add_argument("--config", default="./config/config.yaml", help=".yaml file to use for configuring keypoint extraction")
    parser.add_argument("--box_sample_mode", default="box_nn", help="Type of box sampling")
    args = parser.parse_args()

    if 'SVGA_VGPU10' in os.environ:    # this environment variable may exist in VMware virtual machines
        del os.environ['SVGA_VGPU10']  # remove it to launch Open3D visualizer properly

    os.system(f'./build/test_keypoint {args.config}')

    cloud = o3d.io.read_point_cloud("./results/cloud.ply")
    keypoints = o3d.io.read_point_cloud("./results/keypoint.ply")

    if args.num_fps != -1:
        keypoints = keypoints.farthest_point_down_sample(num_samples=args.num_fps)
    if args.num_contract_split != -1:
        keypoints = sample_box_points(keypoints, cloud, args.num_contract_split, args.box_sample_mode)
    print(f"Final number of keypoints: {len(keypoints.points)}")

    o3d.visualization.draw_geometries([cloud, keypoints_to_spheres(keypoints)])

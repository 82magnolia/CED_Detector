#!/usr/bin/env python3
import os
import open3d as o3d
import argparse
from scripts.contract_box_sampling import sample_box_points, sample_hist_points, build_object_graph


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
    parser.add_argument("--num_hist_bins", type=int, default=-1, help="Optional number of height histogram bins to make for filtering keypoints")
    parser.add_argument("--config", default="./config/config.yaml", help=".yaml file to use for configuring keypoint extraction")
    parser.add_argument("--box_sample_mode", default="box_alpha", help="Type of box sampling")
    parser.add_argument("--hist_sample_mode", default="hist_alpha", help="Type of histogram sampling")
    parser.add_argument("--valid_angle_thres", default=150., help="Angle thresholding to use for alphashape-based keypoint filtering", type=float)
    parser.add_argument("--graph_mode", default="nn", help="Mode to use for making object graph")
    parser.add_argument("--init_nn", default=2, help="Number of nearest neighbors to use when creating initial graph", type=int)
    parser.add_argument("--skip_graph", action="store_true", help="Optionally skip object graph generation")
    args = parser.parse_args()

    if 'SVGA_VGPU10' in os.environ:    # this environment variable may exist in VMware virtual machines
        del os.environ['SVGA_VGPU10']  # remove it to launch Open3D visualizer properly

    os.system(f'./build/test_keypoint {args.config}')

    cloud = o3d.io.read_point_cloud("./results/cloud.ply")
    keypoints = o3d.io.read_point_cloud("./results/keypoint.ply")

    if args.num_fps != -1:
        keypoints = keypoints.farthest_point_down_sample(num_samples=args.num_fps)
    if args.num_contract_split != -1:
        keypoints, keypoints_levels = sample_box_points(keypoints, cloud, args.num_contract_split, args.box_sample_mode, args.valid_angle_thres, True)
    if args.num_hist_bins != -1:
        keypoints, keypoints_levels = sample_hist_points(keypoints, cloud, args.num_hist_bins, args.hist_sample_mode, args.valid_angle_thres, True)
    print(f"Final number of keypoints: {len(keypoints.points)}")

    if not args.skip_graph:
        obj_graph = build_object_graph(keypoints, cloud, keypoints_levels, graph_mode=args.graph_mode, init_nn=args.init_nn)
        print("Built object graph!")
        o3d.visualization.draw_geometries([obj_graph, keypoints_to_spheres(keypoints)])

    o3d.visualization.draw_geometries([cloud, keypoints_to_spheres(keypoints)])

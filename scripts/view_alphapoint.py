#!/usr/bin/env python3
import os
import open3d as o3d
import argparse
from scripts.contract_box_sampling import (
    sample_alpha_points,
    sample_alpha_points_multidirectional,
    build_object_graph,
    simplify_graph
)
import numpy as np

def alphapoints_to_spheres(alphapoints):
    spheres = o3d.geometry.TriangleMesh()
    for alphapoint in alphapoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(alphapoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.0, 0.0])
    return spheres


def mutual_nn_pairs(pcd0, pcd1, nn_dist_thres=None):
    pcd0_np = np.asarray(pcd0.points)
    pcd1_np = np.asarray(pcd1.points)
    pcd0_range = np.arange(pcd0_np.shape[0])
    pcd1_range = np.arange(pcd1_np.shape[0])
    dist_mtx = np.linalg.norm(pcd0_np[:, None, :] - pcd1_np[None, :, :], axis=-1)  # (N_0, N_1)

    # Mutual NN assignment (https://gist.github.com/mihaidusmanu/20fd0904b2102acc1330bad9b4badab8)
    match_0_to_1 = dist_mtx.argmin(-1)  # (N_0)
    match_1_to_0 = dist_mtx.argmin(0)  # (N_1)

    if nn_dist_thres is not None:
        valid_matches = (match_1_to_0[match_0_to_1] == pcd0_range) & (dist_mtx.min(-1) < nn_dist_thres)
    else:
        valid_matches = match_1_to_0[match_0_to_1] == pcd0_range

    match_0_idx = pcd0_range[valid_matches]
    match_1_idx = pcd1_range[match_0_to_1[valid_matches]]

    match_pcd0 = o3d.geometry.PointCloud()
    match_pcd0_np = pcd0_np[match_0_idx]
    match_pcd0.points = o3d.utility.Vector3dVector(match_pcd0_np)

    match_pcd1 = o3d.geometry.PointCloud()
    match_pcd1_np = pcd1_np[match_1_idx]
    match_pcd1.points = o3d.utility.Vector3dVector(match_pcd1_np)

    return match_pcd0, match_0_idx, match_pcd1, match_1_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_name", required=True, help="Name of point cloud .ply file to load")
    parser.add_argument("--num_bins", type=int, default=[3], help="Optional number of histogram bins to make for making alphashape slices", nargs="+")
    parser.add_argument("--valid_angle_thres", default=150., help="Angle thresholding to use for alphashape-based keypoint filtering", type=float)
    parser.add_argument("--graph_mode", default="nn_level_prune", help="Mode to use for making object graph")
    parser.add_argument("--init_nn", default=5, help="Number of nearest neighbors to use when creating initial graph", type=int)
    parser.add_argument("--skip_graph", action="store_true", help="Optionally skip object graph generation")
    parser.add_argument("--simplify_graph", action="store_true", help="Optionally simplify object graph")
    parser.add_argument("--target_num_points", default=15, help="Target number of alphapoints to generate (if set, runs height-based selection with various bins)", type=int)
    parser.add_argument("--visualize_contour", action="store_true", help="Visualize contours for each height bin")
    parser.add_argument("--alphapoint_mode", default="multi_dir", help="Type of alpha point extraction algorithm to run")
    parser.add_argument("--ced_config", default="./config/config.yaml", help=".yaml file to use for configuring keypoint extraction")
    args = parser.parse_args()

    models_list = []
    cloud = o3d.io.read_point_cloud(args.pcd_name)

    num_bins_list = args.num_bins  # Number of bins to test before outputting final object graph
    num_iter = len(num_bins_list)

    for it in range(num_iter):
        print(f"Iteration {it}:")

        if args.alphapoint_mode == 'height':
            alphapoints, alphapoints_levels = sample_alpha_points(cloud, num_bins_list[it], args.valid_angle_thres, True, args.visualize_contour)
        elif args.alphapoint_mode == 'multi':
            alphapoints, alphapoints_levels = sample_alpha_points_multidirectional(cloud, num_bins_list[it], args.valid_angle_thres, True, args.visualize_contour)
        elif args.alphapoint_mode == 'ced_multi':
            # Extract CED keypoints
            if 'SVGA_VGPU10' in os.environ:    # this environment variable may exist in VMware virtual machines
                del os.environ['SVGA_VGPU10']  # remove it to launch Open3D visualizer properly
            if it == 0:  # Only run CED at first run
                os.system(f'./build/test_keypoint {args.ced_config} {args.pcd_name}')
            cloud = o3d.io.read_point_cloud("./results/cloud.ply")
            keypoints = o3d.io.read_point_cloud("./results/keypoint.ply")
            
            # Extract alphapoints
            alphapoints, alphapoints_levels = sample_alpha_points_multidirectional(cloud, num_bins_list[it], args.valid_angle_thres, True, args.visualize_contour)
        
            # Keep keypoints which are mutual nearest neighbors with alphapoints
            alphapoints, match_idx, _, _ = mutual_nn_pairs(alphapoints, keypoints)
            alphapoints_levels = alphapoints_levels[match_idx]
        else:
            raise NotImplementedError("Other alphapoint modes not supported")

        if not args.skip_graph:
            obj_graph = build_object_graph(alphapoints, cloud, alphapoints_levels, graph_mode=args.graph_mode, init_nn=args.init_nn)
            print("Built object graph!")
            
            if args.simplify_graph and len(obj_graph.points) >= 5:  # Only simplify if sufficient points exist in graph
                obj_graph = simplify_graph(obj_graph, valid_angle_thres=args.valid_angle_thres)
            print(f"Current number of alphapoints: {len(obj_graph.points)}")
            models_list.append(obj_graph)
            
            if it == num_iter - 1:
                model_point_counts = np.array([np.asarray(m.points).shape[0] for m in models_list])
                best_model_idx = np.abs(model_point_counts - args.target_num_points).argmin()
                best_model = models_list[best_model_idx]
                print(f"Final number of alphapoints: {len(best_model.points)}")
                o3d.visualization.draw_geometries([best_model, alphapoints_to_spheres(best_model)])
                o3d.visualization.draw_geometries([cloud, alphapoints_to_spheres(best_model)])
                o3d.io.write_line_set("./results/best_model.ply", best_model, write_ascii=True)
        else:
            print(f"Current number of alphapoints: {len(alphapoints.points)}")
            models_list.append(alphapoints)
            if it == num_iter - 1:
                model_point_counts = np.array([np.asarray(m.points).shape[0] for m in models_list])
                best_model_idx = np.abs(model_point_counts - args.target_num_points).argmin()
                best_model = models_list[best_model_idx]
                print(f"Final number of alphapoints: {len(best_model.points)}")
                o3d.visualization.draw_geometries([cloud, alphapoints_to_spheres(best_model)])
                o3d.io.write_point_cloud("./results/best_model.ply", best_model, write_ascii=True)

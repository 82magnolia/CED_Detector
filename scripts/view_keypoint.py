#!/usr/bin/env python3
import os
import open3d as o3d
import argparse
from scripts.contract_box_sampling import (
    sample_box_points,
    sample_hist_points,
    sample_hist_points_multidirectional,
    build_object_graph,
    simplify_graph
)
import numpy as np

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
    parser.add_argument("--pcd_name", required=True, help="Name of point cloud .ply file to load")
    parser.add_argument("--num_fps", type=int, default=15, help="Optional number of points to keep after farthest point sampling")
    parser.add_argument("--num_contract_split", type=int, default=4, help="Optional number of height splits to make for keeping keypoints near bounding boxes")
    parser.add_argument("--num_hist_bins", type=int, default=3, help="Optional number of height histogram bins to make for filtering keypoints")
    parser.add_argument("--keypoints_filter", default="height_bins", help="Type of filtering to first apply to initial keypoints")
    parser.add_argument("--ced_config", default="./config/config.yaml", help=".yaml file to use for configuring keypoint extraction")
    parser.add_argument("--box_sample_mode", default="box_alpha", help="Type of box sampling")
    parser.add_argument("--hist_sample_mode", default="hist_alpha", help="Type of histogram sampling")
    parser.add_argument("--valid_angle_thres", default=150., help="Angle thresholding to use for alphashape-based keypoint filtering", type=float)
    parser.add_argument("--graph_mode", default="nn_level_prune", help="Mode to use for making object graph")
    parser.add_argument("--init_nn", default=5, help="Number of nearest neighbors to use when creating initial graph", type=int)
    parser.add_argument("--skip_graph", action="store_true", help="Optionally skip object graph generation")
    parser.add_argument("--simplify_graph", action="store_true", help="Optionally simplify object graph")
    parser.add_argument("--target_num_points", default=15, help="Target number of keypoints to generate (if set, runs height-based selection with various bins)", type=int)
    parser.add_argument("--visualize_contour", action="store_true", help="Visualize contours for each height bin")
    args = parser.parse_args()

    if 'SVGA_VGPU10' in os.environ:    # this environment variable may exist in VMware virtual machines
        del os.environ['SVGA_VGPU10']  # remove it to launch Open3D visualizer properly

    models_list = []
    os.system(f'./build/test_keypoint {args.ced_config} {args.pcd_name}')
    cloud = o3d.io.read_point_cloud("./results/cloud.ply")
    keypoints = o3d.io.read_point_cloud("./results/keypoint.ply")

    if args.target_num_points == -1:
        num_iter = 1
        num_bins_list = [args.num_hist_bins]
        num_splits_list = [args.num_contract_split]
        num_fps_list = [args.num_fps]
    else:
        num_iter = 3
        num_bins_list = [3, 5, 7]  # Number of bins to test before outputting final object graph
        num_splits_list = [2, 3, 4, 5]
        num_fps_list = [10, 15, 20, 25]

    for it in range(num_iter):
        print(f"Iteration {it}:")

        if args.keypoints_filter == 'fps':
            keypoints = keypoints.farthest_point_down_sample(num_samples=num_fps_list[it])
            keypoints_levels = None
        elif args.keypoints_filter == 'contract_box':
            keypoints, keypoints_levels = sample_box_points(keypoints, cloud, num_splits_list[it], args.box_sample_mode, args.valid_angle_thres, True)
        elif args.keypoints_filter == 'height_bins':
            keypoints, keypoints_levels = sample_hist_points(keypoints, cloud, num_bins_list[it], args.hist_sample_mode, args.valid_angle_thres, True, args.visualize_contour)
        elif args.keypoints_filter == 'height_bins_multi':
            keypoints, keypoints_levels = sample_hist_points_multidirectional(keypoints, cloud, num_bins_list[it], args.hist_sample_mode, args.valid_angle_thres, True, args.visualize_contour)
        else:  # No filtering applied
            keypoints_levels = None

        if not args.skip_graph:
            obj_graph = build_object_graph(keypoints, cloud, keypoints_levels, graph_mode=args.graph_mode, init_nn=args.init_nn)
            print("Built object graph!")
            
            if args.simplify_graph and len(obj_graph.points) >= 5:  # Only simplify if sufficient points exist in graph
                obj_graph = simplify_graph(obj_graph, valid_angle_thres=args.valid_angle_thres)
            print(f"Current number of keypoints: {len(obj_graph.points)}")
            models_list.append(obj_graph)
            
            if it == num_iter - 1:
                model_point_counts = np.array([np.asarray(m.points).shape[0] for m in models_list])
                best_model_idx = np.abs(model_point_counts - args.target_num_points).argmin()
                best_model = models_list[best_model_idx]
                print(f"Final number of keypoints: {len(best_model.points)}")
                o3d.visualization.draw_geometries([best_model, keypoints_to_spheres(best_model)])
                o3d.visualization.draw_geometries([cloud, keypoints_to_spheres(best_model)])
                o3d.io.write_line_set("./results/best_model.ply", best_model, write_ascii=True)
        else:
            print(f"Current number of keypoints: {len(keypoints.points)}")
            models_list.append(keypoints)
            if it == num_iter - 1:
                model_point_counts = np.array([np.asarray(m.points).shape[0] for m in models_list])
                best_model_idx = np.abs(model_point_counts - args.target_num_points).argmin()
                best_model = models_list[best_model_idx]
                print(f"Final number of keypoints: {len(best_model.points)}")
                o3d.visualization.draw_geometries([cloud, keypoints_to_spheres(best_model)])
                o3d.io.write_point_cloud("./results/best_model.ply", best_model, write_ascii=True)

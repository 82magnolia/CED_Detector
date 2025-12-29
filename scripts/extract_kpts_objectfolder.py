import trimesh
import argparse
import numpy as np
import os
from tqdm import tqdm
import open3d as o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectfoloder_root", help="Root directory of ObjectFolder meshes", default="./data/ObjectFolder/")
    parser.add_argument("--save_root", help="Root directory for saving keypoints", default="./data/ObjectFolder_sample_points/")
    parser.add_argument("--mesh_sample_points", default=1000000, help="Number of points to sample from mesh")
    parser.add_argument("--tmp_store_name", default="tmp.ply", help="Temporary file name for storing float-converted .ply to use for CED")
    parser.add_argument("--ced3d_config", default="./config/config_object_folder_3d.yaml", help=".yaml file to use for configuring 3D keypoint extraction")
    parser.add_argument("--ced6d_config", default="./config/config_object_folder_6d.yaml", help=".yaml file to use for configuring 6D keypoint extraction")
    parser.add_argument("--ced_save_dir", default="./results/", help="Directory for temporarily saving CED keypoint extraction results")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
        os.makedirs(os.path.join(args.save_root, "keypoints"), exist_ok=True)
        os.makedirs(os.path.join(args.save_root, "randpoints"), exist_ok=True)

    if not os.path.exists(args.ced_save_dir):
        os.makedirs(args.ced_save_dir, exist_ok=True)

    # List all meshes to process
    mesh_path_list = [os.path.join(args.objectfoloder_root, dir_path, 'model.obj') for dir_path in sorted(os.listdir(args.objectfoloder_root))]

    for mesh_path in tqdm(mesh_path_list, desc="Keypoint generation"):
        obj_idx = eval(mesh_path.split("/")[-2])
        tr_mesh = trimesh.load(mesh_path, force='mesh', process=False)
        points, faces, colors = trimesh.sample.sample_surface(tr_mesh, args.mesh_sample_points, sample_color=True)
        normals = tr_mesh.face_normals[faces]

        # Compute scale: normalize so that longest axis is length 1
        long_axis = (points.max(axis=0) - points.min(axis=0)).argmax()
        axis_len = points[:, long_axis].max() - points[:, long_axis].min()
        points = points / axis_len

        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        o3d_pcd = o3d.t.geometry.PointCloud(device)
        o3d_pcd.point.positions = o3d.core.Tensor(points, o3d.core.float32, device)

        if colors is not None:
            o3d_pcd.point.colors = o3d.core.Tensor(colors[:, :-1], o3d.core.uint8, device)

        o3d_pcd.point.normals = o3d.core.Tensor(normals, o3d.core.float32, device)

        tmp_store_path = os.path.join(args.ced_save_dir, args.tmp_store_name)
        o3d.t.io.write_point_cloud(tmp_store_path, o3d_pcd, compressed=True)

        if colors is not None:
            os.system(f'./build/test_keypoint {args.ced6d_config} {tmp_store_path} {args.ced_save_dir}')
        else:
            os.system(f'./build/test_keypoint {args.ced3d_config} {tmp_store_path} {args.ced_save_dir}')

        os.system(f'rm -rf {tmp_store_path}')

        cloud = o3d.io.read_point_cloud(os.path.join(args.ced_save_dir, "cloud.ply"))
        keypoints = o3d.io.read_point_cloud(os.path.join(args.ced_save_dir, "keypoint.ply"))

        # Re-normalize points
        cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points) * axis_len)
        keypoints.points = o3d.utility.Vector3dVector(np.asarray(keypoints.points) * axis_len)

        # Randomly sample points as equal number as keypoints
        randpoints = cloud.farthest_point_down_sample(num_samples=len(keypoints.points))

        o3d.io.write_point_cloud(os.path.join(args.save_root, "keypoints", f"kpts_{obj_idx}.ply"), keypoints)
        o3d.io.write_point_cloud(os.path.join(args.save_root, "randpoints", f"rpts_{obj_idx}.ply"), randpoints)

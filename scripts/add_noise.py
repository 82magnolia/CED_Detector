import open3d as o3d
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_file", help="Target point cloud .ply file for adding noise")
    parser.add_argument("--noise_sigma", help="Sigma value of point cloud noise", default=0.01, type=float)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd_file)
    pcd_np = np.asarray(pcd.points)
    noise = np.random.randn(*pcd_np.shape) * args.noise_sigma
    pcd_np = pcd_np + noise
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.io.write_point_cloud(args.pcd_file.replace(".ply", f"_noise_{args.noise_sigma}.ply"), pcd, write_ascii=True)

import open3d as o3d
import numpy as np
import alphashape


def generate_contract_box_points(model, num_split):
    points = np.asarray(model.points)

    # Assume points are in canonical space
    bbox_max = np.max(points, axis=0)
    bbox_min = np.min(points, axis=0)
    bbox = np.stack([bbox_min, bbox_max], axis=1)  # (3, 2)

    # Make splits according to y-axis values
    y_points = np.linspace(bbox[1, 0], bbox[1, 1], num_split + 1)
    y_eps = (bbox[1, 1] - bbox[1, 0]) / num_split

    box_point = []
    for idx, y_point in enumerate(y_points[:-1]):
        y_inliers = points[(points[:, 1] < y_point + y_eps) & (points[:, 1] >= y_point)]
        xz_inliers = y_inliers[:, [0, 2]]

        curr_bbox_min = np.min(xz_inliers, axis=0)
        curr_bbox_max = np.max(xz_inliers, axis=0)
        curr_bbox = np.stack([curr_bbox_min, curr_bbox_max], axis=1)  # (2, 2)

        curr_bbox_points = np.array([
            [curr_bbox[0, 0], y_point, curr_bbox[1, 0]],
            [curr_bbox[0, 1], y_point, curr_bbox[1, 0]],
            [curr_bbox[0, 1], y_point, curr_bbox[1, 1]],
            [curr_bbox[0, 0], y_point, curr_bbox[1, 1]]
        ])
        box_point.append(curr_bbox_points)
        next_bbox_points = np.array([
            [curr_bbox[0, 0], y_points[idx+1], curr_bbox[1, 0]],
            [curr_bbox[0, 1], y_points[idx+1], curr_bbox[1, 0]],
            [curr_bbox[0, 1], y_points[idx+1], curr_bbox[1, 1]],
            [curr_bbox[0, 0], y_points[idx+1], curr_bbox[1, 1]]
        ])
        box_point.append(next_bbox_points)

    # Merge box points
    merge_flags = np.zeros(len(box_point) // 2, dtype=bool)  # Flags for whether to merge each box
    merge_thres = 0.01
    for bbox_level in range(len(box_point) // 2 - 1):  # Exclude last index
        curr_level_2d = box_point[bbox_level * 2][:, [0, 2]]
        next_level_2d = box_point[(bbox_level + 1) * 2][:, [0, 2]]

        if np.abs(curr_level_2d - next_level_2d).mean() < merge_thres:
            merge_flags[bbox_level] = True
    merged_box_point = []
    curr_box_point = []

    for flag_idx, flag in enumerate(merge_flags):
        curr_box_point.append(box_point[flag_idx * 2])
        curr_box_point.append(box_point[flag_idx * 2 + 1])
        if not flag:  # Current box and next box does not have overlaps
            agg_box_point = np.stack(curr_box_point, axis=2)  # (4, 3, N)
            avg_box_point = agg_box_point.mean(2, keepdims=True)  # (4, 3, 1)
            agg_box_point_xz = avg_box_point.squeeze() + np.abs(agg_box_point - avg_box_point).max(2) * np.sign(agg_box_point - avg_box_point).mean(2)  # (4, 3)

            merged_bbox_xz = agg_box_point_xz[:, [0, 2]]  # (4, 2)
            merged_bbox_min_y = agg_box_point.min(2)[:, 1:2]  # (4, 1)
            merged_bbox_max_y = agg_box_point.max(2)[:, 1:2]  # (4, 1)
            merged_bbox_min = np.concatenate([merged_bbox_xz[:, 0:1], merged_bbox_min_y, merged_bbox_xz[:, 1:2]], axis=1)  # (4, 3)
            merged_bbox_max = np.concatenate([merged_bbox_xz[:, 0:1], merged_bbox_max_y, merged_bbox_xz[:, 1:2]], axis=1)  # (4, 3)

            merged_box_point.append(merged_bbox_min)
            merged_box_point.append(merged_bbox_max)

            curr_box_point = []
    merged_box_point = np.concatenate(merged_box_point, axis=0)
    merged_box_pcd = o3d.geometry.PointCloud()
    merged_box_pcd.points = o3d.utility.Vector3dVector(merged_box_point)
    merged_box_pcd.paint_uniform_color((0., 0., 1.))

    return merged_box_pcd


def sample_box_points(key_pcd, full_pcd, num_split, sample_mode='box_nn', valid_angle_thres=0., return_level=False):  # Sample points near multi-height boxes
    # key_pcd stores keypoints and full_pcd stores the full point cloud
    merged_box_pcd = generate_contract_box_points(full_pcd, num_split)
    merged_box_np = np.asarray(merged_box_pcd.points)
    if sample_mode == 'box_nn':
        key_np = np.asarray(key_pcd.points)
        dists = np.linalg.norm(merged_box_np[:, None, :] - key_np[None, :, :], axis=-1)
        key_pcd = key_pcd.select_by_index(dists.argmin(1))
        key_levels = None  # Level array not implemented for nearest neighbor case
    elif sample_mode == 'box_height':
        merged_y_points = np.unique(merged_box_np[:, 1])
        key_np = np.asarray(key_pcd.points)
        inlier_y_thres = 0.05
        inlier_y_points = []
        key_levels = []
        for idx, y_point in enumerate(merged_y_points):
            if idx == 0 or idx == len(merged_y_points) - 1:  # Double threshold for top & bottom points
                inlier_np = key_np[(key_np[:, 1] < y_point + inlier_y_thres * 2) & (key_np[:, 1] >= y_point - inlier_y_thres * 2)]
            else:
                inlier_np = key_np[(key_np[:, 1] < y_point + inlier_y_thres) & (key_np[:, 1] >= y_point - inlier_y_thres)]
            inlier_y_points.append(inlier_np)
            key_levels.append(np.ones_like(inlier_np[:, 0], dtype=int) * idx)
        inlier_y_points = np.concatenate(inlier_y_points, axis=0)
        key_pcd = o3d.geometry.PointCloud()
        key_pcd.points = o3d.utility.Vector3dVector(inlier_y_points)
        key_pcd.paint_uniform_color((1., 0., 0.))
        key_levels = np.concatenate(key_levels, axis=0)
    elif sample_mode == 'box_alpha':
        merged_y_points = np.unique(merged_box_np[:, 1])
        key_np = np.asarray(key_pcd.points)
        inlier_y_thres = 0.05
        contour_points = []
        key_levels = []
        for idx, y_point in enumerate(merged_y_points):
            if idx == 0 or idx == len(merged_y_points) - 1:  # Double threshold for top & bottom points
                inlier_np = key_np[(key_np[:, 1] < y_point + inlier_y_thres * 2) & (key_np[:, 1] >= y_point - inlier_y_thres * 2)]
            else:
                inlier_np = key_np[(key_np[:, 1] < y_point + inlier_y_thres) & (key_np[:, 1] >= y_point - inlier_y_thres)]
            inlier_np_xz = inlier_np[:, [0, 2]]
            alpha_shape = alphashape.alphashape(inlier_np_xz, 2.0)  # First attempt with concave hull

            if not alpha_shape.is_empty:
                if alpha_shape.geom_type == 'Polygon':  # Concave hull success
                    contour_xz = np.stack([alpha_shape.exterior.coords.xy[0], alpha_shape.exterior.coords.xy[1]], axis=1)
                    contour_np = np.zeros([contour_xz.shape[0], 3])
                    contour_np[:, [0, 2]] = contour_xz
                    contour_np[:, 1] = y_point
                    contour_np = contour_np[:-1]  # Last point is equal to first point in alphashapes

                    if valid_angle_thres > 0.:
                        # Remove points on lines
                        diff_to_next = contour_np - np.roll(contour_np, 1, axis=0)
                        diff_to_prev = contour_np - np.roll(contour_np, -1, axis=0)
                        diff_to_next = diff_to_next / np.linalg.norm(diff_to_next, axis=-1, keepdims=True)
                        diff_to_prev = diff_to_prev / np.linalg.norm(diff_to_prev, axis=-1, keepdims=True)
                        diff_angle = np.rad2deg(np.arccos((diff_to_next * diff_to_prev).sum(axis=-1)))
                        contour_np = contour_np[diff_angle < valid_angle_thres]
                    
                    dists = np.linalg.norm(contour_np[:, None, :] - key_np[None, :, :], axis=-1)
                    contour_np = key_np[dists.argmin(1)]
                    contour_points.append(contour_np)
                else:
                    alpha_shape = alphashape.alphashape(inlier_np_xz, 0.0)  # Second attempt with convex hull
                    if alpha_shape.geom_type == 'Polygon':  # Convex hull success
                        contour_xz = np.stack([alpha_shape.exterior.coords.xy[0], alpha_shape.exterior.coords.xy[1]], axis=1)
                        contour_np = np.zeros([contour_xz.shape[0], 3])
                        contour_np[:, [0, 2]] = contour_xz
                        contour_np[:, 1] = y_point
                        contour_np = contour_np[:-1]  # Last point is equal to first point in alphashapes

                        if valid_angle_thres > 0.:
                            # Remove points on lines
                            diff_to_next = contour_np - np.roll(contour_np, 1, axis=0)
                            diff_to_prev = contour_np - np.roll(contour_np, -1, axis=0)
                            diff_to_next = diff_to_next / np.linalg.norm(diff_to_next, axis=-1, keepdims=True)
                            diff_to_prev = diff_to_prev / np.linalg.norm(diff_to_prev, axis=-1, keepdims=True)
                            diff_angle = np.rad2deg(np.arccos((diff_to_next * diff_to_prev).sum(axis=-1)))
                            contour_np = contour_np[diff_angle < valid_angle_thres]

                        dists = np.linalg.norm(contour_np[:, None, :] - key_np[None, :, :], axis=-1)
                        contour_np = key_np[dists.argmin(1)]
                        contour_points.append(contour_np)
                    elif alpha_shape.geom_type == 'Point':
                        contour_xz = np.stack([alpha_shape.xy[0], alpha_shape.xy[1]], axis=1)
                        contour_np = np.zeros([contour_xz.shape[0], 3])
                        contour_np[:, [0, 2]] = contour_xz
                        contour_np[:, 1] = y_point

                        dists = np.linalg.norm(contour_np[:, None, :] - key_np[None, :, :], axis=-1)
                        contour_np = key_np[dists.argmin(1)]
                        contour_points.append(contour_np)
                    else:
                        contour_points.append(inlier_np)
            else:
                contour_points.append(inlier_np)
            key_levels.append(np.ones_like(contour_points[-1][:, 0], dtype=int) * idx)

        contour_points = np.concatenate(contour_points, axis=0)
        key_pcd = o3d.geometry.PointCloud()
        key_pcd.points = o3d.utility.Vector3dVector(contour_points)
        key_pcd.paint_uniform_color((1., 0., 0.))
        key_levels = np.concatenate(key_levels, axis=0)
    else:
        raise NotImplementedError("Other sampling modes not supported")

    if return_level:
        return key_pcd, key_levels
    else:
        return key_pcd


def build_object_graph(key_pcd, full_pcd, key_levels=None, graph_mode='nn', init_nn=2, inter_level_init_nn=1):
    key_np = np.asarray(key_pcd.points)
    dists = np.linalg.norm(key_np[:, None, :] - key_np[None, :, :], axis=-1)
    dists[np.diag_indices(key_np.shape[0])] = np.inf
    if graph_mode == 'nn':
        graph_model = o3d.geometry.LineSet()
        graph_model.points = key_pcd.points
        topk_idx = np.argsort(dists, axis=1)[:, :init_nn]
        ref_idx = np.arange(key_np.shape[0])[:, None].repeat(init_nn, axis=1)
        topk_pair = np.stack([ref_idx, topk_idx], axis=-1)  # (N_pts, N_nn, 2)
        topk_pair = topk_pair.reshape(-1, 2)
        line_idx = topk_pair.tolist()
        graph_model.lines = o3d.utility.Vector2iVector(line_idx)
    elif graph_mode in ['nn_level', 'nn_level_prune']:
        graph_model = o3d.geometry.LineSet()
        graph_model.points = key_pcd.points
        assert key_levels is not None
        max_level = key_levels.max()
        graph_lines = []
        for level in range(0, max_level + 1):
            # Add intra-level connections
            level_idx = np.where(key_levels == level)[0]
            level_init_nn = min(level_idx.shape[0], init_nn)
            level_key_pcd = key_pcd.select_by_index(level_idx)
            level_key_np = np.asarray(level_key_pcd.points)
            level_dists = dists[level_idx, :][:, level_idx]
            topk_idx = np.argsort(level_dists, axis=1)[:, :level_init_nn]
            topk_idx = np.take(level_idx, topk_idx)
            ref_idx = np.arange(level_key_np.shape[0])[:, None].repeat(level_init_nn, axis=1)
            ref_idx = np.take(level_idx, ref_idx)
            topk_pair = np.stack([ref_idx, topk_idx], axis=-1)  # (N_pts, N_nn, 2)
            topk_pair = topk_pair.reshape(-1, 2).tolist()
            graph_lines.extend(list(map(tuple, topk_pair)))

            # Add inter-level connections, both with next & previous levels
            if level != max_level:
                next_idx = np.where(key_levels == level + 1)[0]
                curr_next_dists = dists[level_idx, :][:, next_idx]
                topk_idx = np.argsort(curr_next_dists, axis=1)[:, :inter_level_init_nn]
                topk_idx = np.take(next_idx, topk_idx)
                ref_idx = np.arange(level_key_np.shape[0])[:, None].repeat(inter_level_init_nn, axis=1)
                ref_idx = np.take(level_idx, ref_idx)
                topk_pair = np.stack([ref_idx, topk_idx], axis=-1)  # (N_pts, N_nn, 2)
                topk_pair = topk_pair.reshape(-1, 2).tolist()
                graph_lines.extend(list(map(tuple, topk_pair)))
            if level != 0:
                prev_idx = np.where(key_levels == level - 1)[0]
                curr_prev_dists = dists[level_idx, :][:, prev_idx]
                topk_idx = np.argsort(curr_prev_dists, axis=1)[:, :inter_level_init_nn]
                topk_idx = np.take(prev_idx, topk_idx)
                ref_idx = np.arange(level_key_np.shape[0])[:, None].repeat(inter_level_init_nn, axis=1)
                ref_idx = np.take(level_idx, ref_idx)
                topk_pair = np.stack([ref_idx, topk_idx], axis=-1)  # (N_pts, N_nn, 2)
                topk_pair = topk_pair.reshape(-1, 2).tolist()
                graph_lines.extend(list(map(tuple, topk_pair)))

        graph_lines = list(set(graph_lines))  # Remove overlapping lines
        graph_lines = np.array(graph_lines, dtype=int)

        # Optionally prune lines
        if graph_mode == 'nn_level_prune':
            num_line_steps = 10
            nn_search_size = 0.1
            valid_line_thres = 0.7
            full_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(full_pcd, voxel_size=nn_search_size)
            line_starts = key_np[graph_lines[:, 0], None, :]  # (N_lines, 1, 3)
            line_ends = key_np[graph_lines[:, 1], None, :]  # (N_lines, 1, 3)
            line_steps = np.linspace(0., 1., num_line_steps)[None, :, None]  # (1, N_steps, 1)
            line_points = line_starts * line_steps + line_ends * (1 - line_steps)  # (N_lines, N_steps, 3)
            line_in_vox = full_vox.check_if_included(o3d.utility.Vector3dVector(line_points.reshape(-1, 3)))
            line_in_vox = np.array(line_in_vox).reshape(line_points.shape[0], num_line_steps)  # (N_lines, N_steps)
            valid_lines = line_in_vox.sum(-1).astype(float) / num_line_steps > valid_line_thres
            graph_lines = graph_lines[valid_lines]

        graph_model.lines = o3d.utility.Vector2iVector(graph_lines)
    elif graph_mode == 'alpha_shape':
        alpha_shape = alphashape.alphashape(np.asarray(key_pcd.points), 0.2)
        graph_model = alpha_shape.as_open3d
    return graph_model

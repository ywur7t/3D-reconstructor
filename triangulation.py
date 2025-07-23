from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def dense_point_cloud(points, colors, k=100, distance_threshold=0.9, x=[-1000, 1000], y=[-1000, 1000], z=[-1000, 1000], log_func=None):
    def log(msg):
        if log_func:
            log_func(msg)
    points = np.asarray(points)
    colors = np.asarray(colors)
    min_length = min(len(points), len(colors))
    points = points[:min_length]
    colors = colors[:min_length]
    if len(points) != len(colors):
        log((
            f"Points and colors arrays must have same length. Got {len(points)} points and {len(colors)} colors"))
        raise ValueError(
            f"Points and colors arrays must have same length. Got {len(points)} points and {len(colors)} colors")
    log("Фильтрация по координатам")

    if len(points) > 0:
        min_x, max_x = x
        min_y, max_y = y
        min_z, max_z = z

        log_func("Фильтрация по")
        log_func(f"{x}")
        log_func(f"{y}")
        log_func(f"{z}")

        xyz_filter = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
            (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
        )
        points = points[xyz_filter]
        colors = colors[xyz_filter]
    if len(points) > 0:
        _, unique_indices = np.unique(points, axis=0, return_index=True)
        points = points[unique_indices]
        colors = colors[unique_indices]
    if len(points) > k:
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points))).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distances = distances[:, 1:].mean(axis=1)
        mask = mean_distances < distance_threshold
        points = points[mask]
        colors = colors[mask]
    if len(colors) > 0:
        enhanced_colors = np.clip(
            colors * [1.0, 1.1, 1.2], 0, 255).astype(np.uint8)
    else:
        enhanced_colors = np.empty((0, 3), dtype=np.uint8)
    log("Фильтрация закончена")
    return points, enhanced_colors


def run_sfm_pipeline(K, kp, all_matches, log_func=None):
    def log(str):
        if log_func:
            log_func(str)

    log(f'Запуск реконструкции')
    best_pair = select_best_initial_pair(kp, all_matches)
    camera_poses, point3d_ids, points3d, next_point_id = initial_reconstruction(
        best_pair[0], best_pair[1], K, kp, all_matches, log_func=None)
    print(f'\n\n\n {best_pair[0]} | {best_pair[1]}  \n\n\n')
    total_images = len(kp)
    added = {best_pair[0], best_pair[1]}
    for i in range(total_images):
        if i not in added:
            log(f'Добавление камеры {i} ...')
            result = add_image(K, kp, i, all_matches, camera_poses,
                               point3d_ids, points3d, next_point_id)
            if result is not None:
                camera_poses, point3d_ids, points3d, next_point_id = result
                added.add(i)

    return points3d, camera_poses, point3d_ids


def select_best_initial_pair(kp, all_matches):
    best_pair = (0, 1)
    best_matches = 0
    for (i, j), matches in all_matches.items():
        if len(matches) > best_matches:
            best_matches = len(matches)
            best_pair = (i, j)
    return best_pair


def triangulate_two_views(K, R1, t1, R2, t2, pts1, pts2):
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    return (pts4d[:3] / pts4d[3]).T


def recover_pose_from_essential(K, pts1, pts2):
    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                      threshold=0.5,
                                      prob=0.999,
                                      maxIters=2000)
    inliers = inliers.ravel().astype(bool)
    pts1_in, pts2_in = pts1[inliers], pts2[inliers]
    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, pts1_in, pts2_in, inliers


def initial_reconstruction(i, j, K, kp, all_matches, log_func=None):
    def log(str):
        if log_func:
            log_func(str)

    camera_poses = {}
    point3d_ids = {}
    points3d = []
    next_point_id = 0

    matches_ij = all_matches[(i, j)]
    pts_i = np.float32([kp[i][m.queryIdx].pt for m in matches_ij])
    pts_j = np.float32([kp[j][m.trainIdx].pt for m in matches_ij])

    log(f'Востановление поз...')
    R, t, pts_i_in, pts_j_in, inliers = recover_pose_from_essential(
        K, pts_i, pts_j)

    camera_poses[i] = (np.eye(3), np.zeros((3, 1)))
    camera_poses[j] = (R, t)

    log(f'триангуляция камер {i} | {j} ...')

    pts3d = triangulate_two_views(K, camera_poses[i][0], camera_poses[i][1],
                                  camera_poses[j][0], camera_poses[j][1],
                                  pts_i_in, pts_j_in)

    inlier_matches = np.array(matches_ij)[inliers]
    for idx, m in enumerate(inlier_matches):
        point3d_ids[(i, m.queryIdx)] = next_point_id
        point3d_ids[(j, m.trainIdx)] = next_point_id
        points3d.append(pts3d[idx])
        next_point_id += 1

    return camera_poses, point3d_ids, points3d, next_point_id


def add_image(K, kp, img_idx_new, all_matches, camera_poses, point3d_ids, points3d, next_point_id):
    object_points = []
    image_points = []
    matched_3d_ids = []

    for (i1, i2), matches in all_matches.items():
        if i2 != img_idx_new or i1 not in camera_poses:
            continue
        for m in matches:
            key = (i1, m.queryIdx)
            if key in point3d_ids:
                pt3d = points3d[point3d_ids[key]]
                object_points.append(pt3d)
                image_points.append(kp[img_idx_new][m.trainIdx].pt)
                matched_3d_ids.append((img_idx_new, m.trainIdx))

    if len(object_points) < 6:
        return

    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec

    camera_poses[img_idx_new] = (R, t)

    for other_idx in camera_poses:
        if other_idx == img_idx_new:
            continue
        pair = (other_idx, img_idx_new) if (
            other_idx, img_idx_new) in all_matches else (img_idx_new, other_idx)
        if pair not in all_matches:
            continue

        matches = all_matches[pair]
        if pair[0] == img_idx_new:
            pts1 = np.float32(
                [kp[img_idx_new][m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[other_idx][m.trainIdx].pt for m in matches])
        else:
            pts1 = np.float32([kp[other_idx][m.queryIdx].pt for m in matches])
            pts2 = np.float32(
                [kp[img_idx_new][m.trainIdx].pt for m in matches])

        pts3d = triangulate_two_views(K,
                                      camera_poses[other_idx][0], camera_poses[other_idx][1],
                                      camera_poses[img_idx_new][0], camera_poses[img_idx_new][1],
                                      pts1, pts2)

        for k, m in enumerate(matches):
            key1 = (other_idx, m.queryIdx)
            key2 = (img_idx_new, m.trainIdx)
            if key1 in point3d_ids or key2 in point3d_ids:
                continue
            point3d_ids[key1] = next_point_id
            point3d_ids[key2] = next_point_id
            points3d.append(pts3d[k])
            next_point_id += 1

    return camera_poses, point3d_ids, points3d, next_point_id


def get_point_colors(points2d, image):
    h, w = image.shape[:2]
    colors = []
    for pt in points2d:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            colors.append(image[y, x])
        else:
            colors.append([0, 0, 0])
    return np.array(colors)


def save_point_cloud_ply_with_color(filename, points3d, colors):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points3d)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")

        for p, c in zip(points3d, colors):

            if np.max(colors) <= 1.0:
                r, g, b = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            else:
                r, g, b = (int(c[0]), int(c[1]), int(c[2]))
            f.write("{:.4f} {:.4f} {:.4f} {} {} {}\n".format(
                p[0], p[1], p[2], r, g, b))


# def colorize_points(points3d, point3d_ids, kp, image_paths):
#     point_colors = []
#     images = [cv2.imread(p) for p in image_paths]

#     reverse_map = {}
#     for (img_id, kp_id), pt3d_id in point3d_ids.items():
#         reverse_map.setdefault(pt3d_id, []).append((img_id, kp_id))

#     for pt3d_id in range(len(points3d)):
#         if pt3d_id in reverse_map:
#             img_id, kp_id = reverse_map[pt3d_id][0]
#             pt2d = kp[img_id][kp_id].pt
#             img = images[img_id]
#             color = get_point_colors([pt2d], img)[0]
#         else:
#             color = [0, 0, 0]
#         point_colors.append(color)

#     return np.array(point_colors)
def colorize_points(points3d, point3d_ids, kp, image_paths):
    point_colors = []
    images = [cv2.imread(p) for p in image_paths]
    reverse_map = {}
    for (img_id, kp_id), pt3d_id in point3d_ids.items():
        reverse_map.setdefault(pt3d_id, []).append((img_id, kp_id))
    for pt3d_id in range(len(points3d)):
        color = [0, 0, 0]
        if pt3d_id in reverse_map:
            for img_id, kp_id in reverse_map[pt3d_id]:
                try:
                    if img_id < len(kp) and kp_id < len(kp[img_id]):
                        pt2d = kp[img_id][kp_id].pt
                        img = images[img_id]
                        color = get_point_colors([pt2d], img)[0]
                        break 
                except (IndexError, AttributeError):
                    continue 
        point_colors.append(color)
    return np.array(point_colors)


def duplicate_around_points(points, colors, radius=0.4, points_per_point=5, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0):
    new_points = []
    new_colors = []

    for i in tqdm(range(len(points)), desc="Расширяем точки в Y ∈ [20, 30]"):
        center = points[i]
        if (y_min <= center[1] <= y_max) and (x_min <= center[0] <= x_max) and (z_min <= center[2] <= z_max):
            color = colors[i]

            rand_dirs = np.random.normal(0, 1, (points_per_point, 3))
            rand_dirs /= np.linalg.norm(rand_dirs, axis=1)[:, None]
            rand_radii = np.random.uniform(
                0, radius, points_per_point)[:, None]
            offsets = rand_dirs * rand_radii
            new_pts = center + offsets
            new_points.extend(new_pts)
            new_colors.extend([color] * points_per_point)

    all_points = np.vstack((points, new_points))
    all_colors = np.vstack((colors, new_colors))

    dense_pcd = o3d.geometry.PointCloud()
    dense_pcd.points = o3d.utility.Vector3dVector(all_points)
    dense_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    return dense_pcd

from tqdm import tqdm
import cv2
import numpy as np


def write_ply_from_matches(all_matches, kp, image_paths, output_file="keypoints/matches.ply", log_func=None):
    if log_func:
        log_func(f"Сохранение point cloud в PLY файл")
    images = [cv2.imread(path) for path in image_paths]
    points = []
    colors = []
    z = 0
    used_indices = set()
    for (i, j), matches in tqdm(all_matches.items(), desc="Обработка совпадений"):
        z += 100
        for match in matches:
            if isinstance(match, cv2.DMatch):
                idx1 = match.queryIdx
                idx2 = match.trainIdx
            elif isinstance(match, tuple) and len(match) >= 2:
                idx1 = match[0]
                idx2 = match[1]
            else:
                continue
            if (i, idx1) not in used_indices:
                if idx1 < len(kp[i]):
                    x, y = map(int, kp[i][idx1].pt)
                    if 0 <= x < images[i].shape[1] and 0 <= y < images[i].shape[0]:
                        color = images[i][y, x][::-1]
                        points.append([x, y, z])
                        colors.append(color)
                        used_indices.add((i, idx1))
            if (j, idx2) not in used_indices:
                if idx2 < len(kp[j]):
                    x, y = map(int, kp[j][idx2].pt)
                    if 0 <= x < images[j].shape[1] and 0 <= y < images[j].shape[0]:
                        color = images[j][y, x][::-1]
                        points.append([x, y, z])

                        colors.append(color)
                        used_indices.add((j, idx2))
    if not points:
        if log_func:
            log_func("Нет точек для сохранения в PLY файл")
        return

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    with open(output_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, col in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")

    if log_func:
        log_func(
            f"Цветной PLY-файл сохранён как {output_file} (точек: {len(points)})")


def save_ply(filename, points, colors, color_order='bgr', log_func=None):

    if log_func:
        log_func(f"Сохранение point cloud в PLY файл")
        log_func(
            f"element vertex: points: {len(points)}  colors: {len(colors)}")

    with open(filename, 'w') as f:

        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for p, c in zip(points, colors):
            if color_order.lower() == 'bgr':

                f.write(
                    f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
            else:

                f.write(
                    f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

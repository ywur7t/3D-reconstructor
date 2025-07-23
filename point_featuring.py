import cv2
import numpy as np
import os
from error_handle import error_handle


def save_keypoints_to_file(image_path, keypoints, descriptors, output_dir="keypoints"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    ply_file = os.path.join(output_dir, f"{base_name}_keypoints.ply")
    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(keypoints)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for kp in keypoints:
            f.write(f"{kp.pt[0]} {kp.pt[1]} 0\n")

    print(f"Сохранено: {ply_file}")


def Point_Featuring(gray_paths, mask_paths, sift_point, random_point, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, log_func=None):
    if log_func:
        log_func(f"Начало выделения точек...")
    kp = []
    des = []
    min_points = float('inf')
    for mask in mask_paths:
        ys, xs = np.where(mask >= 250)
        min_points = min(min_points, len(xs))
    num_random_points = random_point
    for i in range(len(gray_paths)):
        if log_func:
            log_func(f"Выделение точек: {gray_paths[i]}")
        image = cv2.imread(gray_paths[i])
        mask = mask_paths[i]
        sift = cv2.SIFT_create(
            nfeatures=sift_point,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
        keypoints, descriptors = sift.detectAndCompute(image, mask)
        if log_func:
            log_func(
                f"Изначальные дескрипторы: {len(descriptors) if descriptors is not None else 0}")
        if log_func:
            log_func(f"Генерация случайных точек")
        random_kps = tuple(add_random_points(
            mask, num_points=num_random_points))
        if random_kps:
            _, random_des = sift.compute(image, random_kps)
            if descriptors is not None and random_des is not None:
                keypoints += random_kps
                descriptors = np.vstack([descriptors, random_des])
            elif random_des is not None:
                keypoints = list(random_kps)
                descriptors = random_des
        kp.append(keypoints)
        des.append(descriptors)
        if log_func:
            log_func(
                f"Финальные дескрипторы: {len(des[i]) if des[i] is not None else 0}")
        if log_func:
            log_func(f"Сохранение ключевых точек в файл")
        save_keypoints_to_file(gray_paths[i], keypoints, descriptors)
        output_image = cv2.drawKeypoints(image, keypoints, None)
        cv2.imwrite(
            f'./images/keypoints/keypoints_{os.path.basename(gray_paths[i])}', output_image)
        if log_func:
            log_func(f"Сохранение серого изображения с точками")
    return kp, des


def add_random_points(mask, num_points):
    ys, xs = np.where(mask >= 250)
    coords = list(zip(xs, ys))
    if len(coords) == 0:
        return []

    num_points = num_points
    chosen = np.random.choice(len(coords), size=num_points, replace=True)
    random_points = [cv2.KeyPoint(
        float(coords[i][0]), float(coords[i][1]), 1) for i in chosen]

    return random_points

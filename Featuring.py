import cv2
import numpy as np
import matplotlib.pyplot as plt



def extract_keypoints_with_depth(img, disparity, focal_length, baseline, depth_threshold=(0.1, 5.0)):
    """
    Выделяет ключевые точки с учётом карты глубины.

    **Входные параметры**:
    img – изображение,
    disparity – карта смещений,
    focal_length – фокусное расстояние (в пикселях),
    baseline – расстояние между камерами (в метрах),
    depth_threshold – (min, max) глубины в метрах для фильтрации.

    **Переменные**:
    depth_map – карта глубины,
    keypoints – список ключевых точек,
    valid_keypoints – отфильтрованные точки по глубине,
    descriptors – дескрипторы.

    **Выходные параметры**:
    keypoints, descriptors, depth_map – список точек, их дескрипторы, и карта глубины.
    """

    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / disparity
    depth_map[disparity <= 0.0] = 0


    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(cv2.imread(img, cv2.IMREAD_GRAYSCALE), None)

    points_with_depth = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            d = disparity[y, x]
            if d > 0:
                Z = (focal_length * baseline) / d * 50 * 1000

                points_with_depth.append((x, y, Z))

    return keypoints, descriptors, points_with_depth


# def extract_keypoints_with_depth(img, disparity, focal_length, baseline, depth_threshold=(0.1, 5.0)):
#     """
#     Выделяет ключевые точки с учётом карты глубины и добавляет точки границ.

#     **Входные параметры**: img, disparity, focal_length, baseline, depth_threshold – см. выше.
#     **Переменные**: depth_map, keypoints, descriptors, edge_keypoints, valid_keypoints.
#     **Выходные параметры**: keypoints, descriptors, points_with_depth – SIFT + границы.
#     """
#     with np.errstate(divide='ignore'):
#         depth_map = (focal_length * baseline) / disparity
#     depth_map[disparity <= 0.0] = 0

#     gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(gray, None)

#     # === Добавление точек с границ ===
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     edge_magnitude = cv2.magnitude(sobel_x, sobel_y)
#     edge_magnitude = cv2.convertScaleAbs(edge_magnitude)

#     # Пороговая фильтрация, чтобы не брать шум
#     _, edge_binary = cv2.threshold(edge_magnitude, 100, 255, cv2.THRESH_BINARY)

#     # Ищем координаты ненулевых точек
#     edge_coords = np.column_stack(np.where(edge_binary > 0))

#     # Преобразуем координаты в keypoints
#     edge_keypoints = [cv2.KeyPoint(float(x), float(y), 1) for y, x in edge_coords]

#     # Объединяем SIFT + границы
#     all_keypoints = list(keypoints) + edge_keypoints
#     all_keypoints, descriptors = sift.compute(gray, all_keypoints)  # Перерасчёт дескрипторов

#     # === Добавление глубины для всех точек ===
#     points_with_depth = []
#     for kp in all_keypoints:
#         x, y = int(kp.pt[0]), int(kp.pt[1])
#         if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
#             d = disparity[y, x]
#             if d > 0:
#                 Z = (focal_length * baseline) / d
#                 if depth_threshold[0] <= Z <= depth_threshold[1]:
#                     points_with_depth.append((x, y, Z))

#     return all_keypoints, descriptors, points_with_depth











def draw_keypoints_with_depth(img, points_with_depth):

    scale_z = 50
    xs, ys, zs = zip(*[(x, y, z * scale_z) for x, y, z in points_with_depth])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c=zs, cmap='jet', s=3)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (depth)')

    ax.view_init(elev=20, azim=-60)
    plt.title("SIFT Keypoints in 3D with Depth")
    plt.show()

from collections import defaultdict
import cv2
import numpy as np
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import os
from error_handle import error_handle


def br_force_optimized(kp, des, image_paths=None, visualize=True, max_workers=4, good_distance=0.75, ransacReprojThreshold=1.5, maxIters=5000, log_func=None):
    images, gray_images = [], []
    if image_paths is not None:
        for path in image_paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            gray_images.append(gray)
    else:
        gray_images = [None] * len(kp)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    all_matches = {}
    if log_func:
        log_func(f"Создание пар")
    pairs = list(combinations(range(len(des)), 2))
    if log_func:
        log_func(f"Пары созданы: {pairs}")

    def process_pair(i, j):
        if log_func:
            log_func(f"Обработка пары {(i, j)}")
        desc1, desc2 = des[i], des[j]
        if desc1 is None or desc2 is None or desc1.shape[1] != desc2.shape[1]:
            return (i, j), []
        matches = flann.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance <
                good_distance * n.distance]
        if len(good) >= 8:
            pts1 = np.float32([kp[i][m.queryIdx].pt for m in good])
            pts2 = np.float32([kp[j][m.trainIdx].pt for m in good])

            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=ransacReprojThreshold,
                confidence=0.99,
                maxIters=maxIters
            )
            H, H_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)

            mask = np.logical_or(mask.ravel(), H_mask.ravel())

            if mask is not None and mask.sum() > 0:

                inliers = [good[k] for k in range(len(good)) if mask[k]]

                if visualize and (gray_images[i] is not None) and (gray_images[j] is not None):
                    if log_func:
                        log_func(f"сохранение match_{i}_{j}.jpg")
                    img_matches = cv2.drawMatches(
                        gray_images[i], kp[i],
                        gray_images[j], kp[j],
                        inliers, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    save_path = f"./images/matches/match_{i}_{j}.jpg"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, img_matches)
                return (i, j), inliers
        return (i, j), []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: process_pair(*p), pairs))
    all_matches = dict(results)
    return all_matches


def build_tracks(matches_dict, log_func=None):

    if log_func:
        log_func(f"Построение графа")
    parent = dict()
    track_map = defaultdict(list)

    def find(u):

        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru

    for (i, j), matches in matches_dict.items():
        if log_func:
            log_func(f"пара ({i} ; {j})")

        for idx_i, idx_j in matches:
            u = (i, idx_i)
            v = (j, idx_j)
            if u not in parent:
                parent[u] = u
            if v not in parent:
                parent[v] = v
            union(u, v)

    if log_func:
        log_func(f"Собираем компоненты связности")
    for key in parent:
        if log_func:
            log_func(f"Ключ {key}")
        root = find(key)
        track_map[root].append(key)

    if log_func:
        log_func(f"Перенумерация track id")
    numbered_tracks = {}
    for i, obs in enumerate(track_map.values()):
        if len(obs) >= 2:
            numbered_tracks[i] = obs

    return numbered_tracks

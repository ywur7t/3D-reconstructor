import cv2
import numpy as np
from typing import List
from error_handle import error_handle


def check_image_count(image_paths: List[str]) -> bool:
    return len(image_paths) >= 2


def check_empty_paths(image_paths: List[str]) -> bool:
    return len(image_paths) == 0


def check_empty_image(image_path: str) -> bool:
    try:
        img = cv2.imread(image_path)
        return img is None or img.size == 0
    except:
        return True


def check_resolution(image_path: str) -> bool:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        min_w, min_h = 720, 480
        max_w, max_h = 1920, 1080
        return (min_w <= w <= max_w) and (min_h <= h <= max_h)
    except:
        return False


def check_color_depth(image_path: str) -> bool:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        return img.dtype == np.uint8
    except:
        return False


def check_noise_level(image_path: str, threshold: float = 2.0) -> bool:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        noise_percent = (100 - (laplacian_var / 10))
        return noise_percent <= threshold
    except:
        return False


def check_distortion(image_path: str, threshold: float = 0.5) -> bool:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        ideal_aspect = 16/9
        current_aspect = w / h
        distortion = abs(current_aspect - ideal_aspect) / ideal_aspect * 100
        return distortion <= threshold
    except:
        return False


def validate_all_images(image_paths: List[str]) -> dict:
    results = {
        'count_ok': check_image_count(image_paths),
        'empty_paths': check_empty_paths(image_paths),
        'images': []
    }
    if not results['empty_paths']:
        for path in image_paths:
            img_results = {
                'path': path,
                'empty': check_empty_image(path),
                'resolution_ok': check_resolution(path),
                'color_depth_ok': check_color_depth(path),
                'noise_ok': check_noise_level(path),
                'distortion_ok': check_distortion(path),
                'all_ok': False
            }
            img_results['all_ok'] = (
                not img_results['empty'] and
                img_results['resolution_ok'] and
                img_results['color_depth_ok'] and
                img_results['noise_ok'] and
                img_results['distortion_ok']
            )
            results['images'].append(img_results)
    return results

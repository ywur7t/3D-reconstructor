import cv2
import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance
from error_handle import error_handle


def Process_Images(image_paths: list,
                   bright: float = 1.0,
                   contrast: float = 1.0,
                   color_balance: tuple = (1.0, 1.0, 1.0),
                   hist_normalize: bool = False,
                   sharp_radius: float = 1.0,
                   sharp_percent: float = 150,
                   sharp_threshold: int = 3,
                   denoise_strength: float = 10.0,
                   denoise_template: int = 7,
                   denoise_search: int = 21,
                   log_func=None) -> list:

    if log_func:
        log_func("Начало обработки изображений...")

    os.makedirs("images/processed", exist_ok=True)
    processed_paths = []

    for image_path in image_paths:
        if log_func:
            log_func(f"Обработка изображения {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            if log_func:
                log_func(
                    f"Ошибка: не удалось прочитать изображение {image_path}")
            continue

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if log_func:
            log_func("Применение коррекции цвета...")
        r, g, b = img_pil.split()
        r = r.point(lambda i: i * color_balance[0])
        g = g.point(lambda i: i * color_balance[1])
        b = b.point(lambda i: i * color_balance[2])
        img_pil = Image.merge('RGB', (r, g, b))

        if log_func:
            log_func("Настройка яркости и контраста...")
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(bright)

        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast)

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if hist_normalize:
            if log_func:
                log_func("Применение нормализации гистограммы...")
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        if denoise_strength > 0:
            if log_func:
                log_func("Устранение шумов...")
            img = cv2.fastNlMeansDenoisingColored(img,
                                                  None,
                                                  h=denoise_strength,
                                                  templateWindowSize=denoise_template,
                                                  searchWindowSize=denoise_search)

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if log_func:
            log_func("Коррекция резкости...")
        img_pil = img_pil.filter(ImageFilter.UnsharpMask(
            radius=sharp_radius,
            percent=sharp_percent,
            threshold=sharp_threshold
        ))

        output_path = os.path.join(
            "images/processed", f"processed_{os.path.basename(image_path)}")
        img_pil.save(output_path)

        if log_func:
            log_func(f"Улучшенное изображение сохранено: {output_path}")

        processed_paths.append(output_path)

    return processed_paths


def GrayScale_Images(image_paths, noise_checker, canny_threshold1, canny_threshold2, kernel_size, clahe_clip_limit, clahe_grid_size,
                     noise_mean, noise_stddev, blur_kernel_size, blur_sigma, log_func=None):

    if log_func:
        log_func(f"Создание изображений в оттенках серого")

    grayscale_paths = []
    mask_paths = []
    output_dir = "images/grayscales"
    os.makedirs(output_dir, exist_ok=True)

    for path in image_paths:

        if log_func:
            log_func(f"Обработка изображения {path}")

        img = cv2.imread(path)
        if img is None:
            continue
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if log_func:
            log_func(f"Создание маски по контуру")

        edges = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_img)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [max_contour], -1,
                             255, thickness=cv2.FILLED)

        if noise_checker == "on":

            if log_func:
                log_func(f"Добавление шума {noise_mean}")

            noise = np.random.normal(
                noise_mean, noise_stddev, gray_img.shape).astype(np.uint8)
            noisy = cv2.add(noise, noise, mask=mask)
        else:
            noisy = gray_img
        if blur_kernel_size != 0 and blur_sigma != 0:

            if log_func:
                log_func(
                    f"Добавление размытия {blur_kernel_size} {blur_sigma}")
            noisy_blurred = cv2.GaussianBlur(
                noisy, (blur_kernel_size, blur_kernel_size), blur_sigma)
        else:
            noisy_blurred = noisy

        if log_func:
            log_func(f"Добавление контраста")

        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(
            clahe_grid_size, clahe_grid_size))
        enhanced_gray = clahe.apply(noisy_blurred)

        if log_func:
            log_func(f"Применение маски")

        enhanced_gray = cv2.bitwise_and(enhanced_gray, mask)

        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, enhanced_gray)

        if log_func:
            log_func(f"Сохранение серого изображения")

        grayscale_paths.append(save_path)
        mask_paths.append(mask)

    return grayscale_paths, mask_paths

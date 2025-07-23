import os
import shutil
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import cv2
import numpy as np
from writing import write_ply_from_matches, save_ply

import customtkinter as ctk
import os
from threading import Thread

from process_images import Process_Images, GrayScale_Images
from point_featuring import Point_Featuring
from points_matching import br_force_optimized
from triangulation import colorize_points, save_point_cloud_ply_with_color, duplicate_around_points, run_sfm_pipeline
from reconstruct import read_ply_ascii_with_colors, from_mesh_to_pointcloud, reconstruct_surface_from_point_cloud
from triangulation import dense_point_cloud
from dataclasses import dataclass


import threading
import time
from image_check import check_empty_image, check_resolution, check_color_depth, check_noise_level, check_distortion, check_image_count, check_empty_paths
from error_handle import error_handle #handle_error


@dataclass
class ImageData:
    paths: list = None
    processed: list = None
    grayscale: list = None
    masks: list = None

    def __post_init__(self):
        self.paths = self.paths or []
        self.processed = self.processed or []
        self.grayscale = self.grayscale or []
        self.masks = self.masks or []

@dataclass
class FeatureData:
    keypoints: any = None
    descriptors: any = None
    matches: any = None
    point_cloud = {
        'points': None,
        'colors': None
    }
    mesh: any = None

def log_message(self, message):
    self.debug_text.configure(state="normal")
    self.debug_text.insert("end", message + "\n")
    self.debug_text.configure(state="disabled")
    self.debug_text.see("end")
    self.update()

    log_file = "notes/debug_log.txt"
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(message + "\n")


class ProgressWindow(ctk.CTkToplevel):
    def __init__(self, parent, total_images):
        super().__init__(parent)
        self.title("Проверка изображений")
        self.geometry("400x150")
        self.resizable(False, False)

        self.total_images = total_images
        self.current = 0

        self.progress_label = ctk.CTkLabel(self, text="Подготовка к проверке...")
        self.progress_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self, width=350)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.pack(pady=5)

        self.grab_set()

    def update_progress(self, current, status):
        self.current = current
        progress = current / self.total_images
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"Проверяется изображение {current} из {self.total_images}")
        self.status_label.configure(text=status)
        self.update()

class ResultsWindow(ctk.CTkToplevel):
    def __init__(self, parent, results):
        super().__init__(parent)
        self.title("Результаты проверки")
        self.geometry("600x500")
        self.results = results
        self.create_widgets()
        self.display_results()

    def create_widgets(self):
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.summary_tab = self.notebook.add("Общая информация")
        self.details_tab = self.notebook.add("Детальная информация")
        self.close_btn = ctk.CTkButton(self, text="Закрыть", command=self.destroy)
        self.close_btn.pack(pady=10)

    def display_results(self):
        summary_frame = ctk.CTkFrame(self.summary_tab)
        summary_frame.pack(fill="both", expand=True, padx=10, pady=10)
        count_ok = "Количество допустимо" if self.results['count_ok'] else "Количество недопустимо"
        ctk.CTkLabel(summary_frame, text=f"Количество изображений: {len(self.results['images'])} {count_ok}",
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=5)
        valid_count = sum(1 for img in self.results['images'] if img['all_ok'])
        ctk.CTkLabel(summary_frame, text=f"Корректных изображений: {valid_count}/{len(self.results['images'])}").pack(anchor="w", pady=5)
        if valid_count==0:
            self.error=3
        else:
            self.error=0
        details_frame = ctk.CTkScrollableFrame(self.details_tab)
        details_frame.pack(fill="both", expand=True, padx=10, pady=10)
        for i, img_result in enumerate(self.results['images'], 1):
            frame = ctk.CTkFrame(details_frame)
            frame.pack(fill="x", pady=5, padx=5)
            status = "ОК" if img_result['all_ok'] else "Нет"
            ctk.CTkLabel(frame, text=f"Изображение {i}: {img_result['path']} Статус: {status}",
                        font=ctk.CTkFont(weight="bold")).pack(anchor="w")
            details = [
                f"Пустое: {'Да' if img_result['empty'] else 'Нет'}",
                f"Разрешение: {'OK' if img_result['resolution_ok'] else 'Не соответствует'}",
                f"Глубина цвета: {'OK' if img_result['color_depth_ok'] else 'Не соответствует'}",
                f"Уровень шума: {'OK' if img_result['noise_ok'] else 'Превышен'}",
                f"Дисторсия: {'OK' if img_result['distortion_ok'] else 'Превышена'}"
            ]
            for detail in details:
                ctk.CTkLabel(frame, text=detail).pack(anchor="w", padx=20)


class App(ctk.CTk):

    def download_model(self):
        try:
            initial_dir = os.path.abspath(os.path.expanduser("~/Documents"))

            file = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                defaultextension=".ply",
                filetypes=[
                    ("OBJ файлы", "*.obj"),
                    ("PLY файлы", "*.ply")
                ],
                title="Сохранить как"
            )

            if not file:
                return

            if not hasattr(self.features, 'mesh') or self.features.mesh is None:
                self.error = 18
                error_handle(self.error)
                return

            ext = os.path.splitext(file)[1].lower()

            if ext == ".ply":
                o3d.io.write_triangle_mesh(file, self.features.mesh, write_vertex_colors=True, write_vertex_normals=True )
            elif ext == ".obj":
                o3d.io.write_triangle_mesh(file, self.features.mesh, write_vertex_colors=True, write_vertex_normals=True )
            else:
                return

            self.error = 0
            error_handle(self.error)

        except Exception as e:
            self.error = 19
            error_handle(self.error, f"Ошибка при сохранении: {str(e)}")
            import traceback
            traceback.print_exc()


    def load_cloud(self):
        file = filedialog.askopenfilename(
            initialdir=os.path.expanduser("./"),
            defaultextension=".ply",
            filetypes=[
                ("PLY файлы", "*.ply"),
                ("OBJ файлы", "*.obj"),
                ("Все файлы", "*.*")
            ],
            title="открыть"
        )
        self.features.point_cloud['points'], self.features.point_cloud['colors'] = read_ply_ascii_with_colors(file)
        self.features.keypoints = None
        self.features.descriptors = None
        self.features.matches = None
        self.current_image = None
        self.show_visualisation(file)


    def create_left_panel(self):
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.buttons_frame = ctk.CTkFrame(self.left_frame)
        self.buttons_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.load_btn = ctk.CTkButton(
            self.buttons_frame, text="Load Images", command=self.load_images)
        self.load_btn.pack(padx=10, pady=10, fill="x")

        self.download_btn = ctk.CTkButton(
            self.buttons_frame, text="Download 3D Model", command=self.download_model)
        self.download_btn.pack(padx=10, pady=10, fill="x")

        self.load_cloud_button = ctk.CTkButton(
            self.buttons_frame, text="load_cloud", command=self.load_cloud)
        self.load_cloud_button.pack(padx=10, pady=10, fill="x")

        self.debug_frame = ctk.CTkFrame(self.left_frame)
        self.debug_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.debug_text = ctk.CTkTextbox(
            self.debug_frame, wrap="word", state="disabled")
        self.debug_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_central_panel(self):
        self.center_frame = ctk.CTkFrame(self)
        self.center_frame.grid(row=0, column=1, padx=10, pady=0, sticky="nsew")
        self.center_frame.grid_rowconfigure(1, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        self.image_list_frame = ctk.CTkScrollableFrame(self.center_frame, height=200)
        self.image_list_frame.grid(row=0, column=0, pady=(2, 10), sticky="nsew")
        self.image_list_frame.grid_columnconfigure(0, weight=1)

        self.visualization_frame = ctk.CTkScrollableFrame(self.center_frame)
        self.visualization_frame.grid(row=1, column=0, sticky="nsew")
        self.visualization_label = ctk.CTkLabel(self.visualization_frame, text="")
        self.visualization_label.pack(expand=True, fill="both")

    def create_right_panel(self):
        self.right_frame = ctk.CTkScrollableFrame(self)
        self.right_frame.grid(row=0, column=2, padx=2, pady=2, sticky="nsew")
        self.create_right_Image_Improve()
        self.create_right_Grayscale()
        self.create_right_Points()
        self.create_right_Matches()
        self.create_right_Triangulate()
        self.create_right_Filter()
        self.create_right_Reconstruct()

    def create_right_Image_Improve(self):
        ctk.CTkLabel(self.right_frame, text="Улучшение изображений").pack(pady=5)
        self.bright_slider, self.bright_value_label = self.create_slider_with_label(
            self.right_frame, "Яркость", 0, 10, initial=1)
        self.contrast_slider, self.contrast_value_label = self.create_slider_with_label(
            self.right_frame, "Контраст", 0, 50, initial=1)
        self.radius_slider, self.radius_value_label = self.create_slider_with_label(
            self.right_frame, "Радиус пикселей", 1, 10, initial=1, steps=10)
        self.percent_slider, self.percent_value_label = self.create_slider_with_label(
            self.right_frame, "Степень резкости", 1, 200, initial=1, steps=200)
        self.threshold_slider, self.threshold_value_label = self.create_slider_with_label(
            self.right_frame, "Изменение яркости", 1, 5, initial=1, steps=5)

        self.color_balance_x_slider, self.color_balance_x_value_label = self.create_slider_with_label(
            self.right_frame, "Изменение Баланса цвета Х", 0, 255, initial=0)
        self.color_balance_y_slider, self.color_balance_y_value_label = self.create_slider_with_label(
            self.right_frame, "Изменение Баланса цвета Y", 0, 255, initial=0)
        self.color_balance_z_slider, self.color_balance_z_value_label = self.create_slider_with_label(
            self.right_frame, "Изменение Баланса цвета Z", 0, 255, initial=0)

        self.hist_normalize_slider = ctk.CTkSwitch(master=self.right_frame, text="Нормализация Гистограммы",variable=ctk.StringVar(value="off"),onvalue="on",offvalue="off")

        self.denoise_strength_slider, self.denoise_strength_value_label = self.create_slider_with_label(
            self.right_frame, "Устранение шума", 0, 50, initial=0)

        self.denoise_template_slider, self.denoise_template_value_label = self.create_slider_with_label(
            self.right_frame, "Шаблон шума", 0, 50, initial=0, steps=50)
        self.denoise_search_slider, self.denoise_search_value_label = self.create_slider_with_label(
            self.right_frame, "Поиск шума", 0, 50, initial=0, steps=50)

        self.process_btn = ctk.CTkButton(self.right_frame, text="Улучшить", command=self.process_images)
        self.process_btn.pack(pady=20)

    def create_right_Grayscale(self):
        ctk.CTkLabel(self.right_frame,text="Улучшение маски и grayscale").pack(pady=5)
        self.minedge_slider, _ = self.create_slider_with_label(self.right_frame, "Минимум края", 0, 300, initial=0)
        self.maxedge_slider, _ = self.create_slider_with_label(
            self.right_frame, "Максимум края", 0, 300, initial=100)
        self.kernel_size_slider, _ = self.create_slider_with_label(
            self.right_frame, "Размер ядра", 0, 10, initial=5,steps=10)
        self.clahe_clip_limit_slider, _ = self.create_slider_with_label(
            self.right_frame, "clahe_clip_limit", 1, 5, initial=3)
        self.clahe_grid_size_slider, _ = self.create_slider_with_label(
            self.right_frame, "clahe_grid_size", 1, 10, initial=8,steps=10)
        self.noise_checker = ctk.CTkSwitch(master=self.right_frame,
                                           text="Добавить шум",
                                           variable=ctk.StringVar(value="on"),
                                           onvalue="on",
                                           offvalue="off")
        self.noise_checker.pack(pady=20)
        self.noise_mean_slider, _ = self.create_slider_with_label(
            self.right_frame, "noise_mean", 0, 10, initial=0)
        self.noise_stddev_slider, _ = self.create_slider_with_label(
            self.right_frame, "noise_stddev", 0, 20, initial=15)
        self.blur_kernel_size_slider, _ = self.create_slider_with_label(
            self.right_frame, "blur_kernel_size", 1, 15, initial=5, steps=7)
        self.blur_sigma_slider, _ = self.create_slider_with_label(
            self.right_frame, "blur_sigma", 0, 10, initial=0, steps=10)
        self.process_btn = ctk.CTkButton(
            self.right_frame, text="Улучшить", command=self.grayscale_images)
        self.process_btn.pack(pady=20)

    def create_right_Points(self):
        ctk.CTkLabel(self.right_frame, text="Выделение точек").pack(pady=5)
        self.SIFT_slider, _ = self.create_slider_with_label(
            self.right_frame, "SIFT", 1000, 500000, initial=1000, steps=500000-1000+1)
        self.Random_slider, _ = self.create_slider_with_label(
            self.right_frame, "Random", 0, 500000, initial=1000, steps=500000)
        self.nOctaveLayers_slider, _ = self.create_slider_with_label(
            self.right_frame, "nOctaveLayers", 1, 10, initial=3, steps=10)
        self.contrastThreshold_slider, _ = self.create_slider_with_label(
            self.right_frame, "contrastThreshold", 0, 1, initial=0.04)
        self.edgeThreshold_slider, _ = self.create_slider_with_label(
            self.right_frame, "edgeThreshold", 1, 200, initial=10, steps=200)
        self.sigma_slider, _ = self.create_slider_with_label(
            self.right_frame, "sigma", 0, 2, initial=1.6)
        self.process_btn = ctk.CTkButton(
            self.right_frame, text="Выделить", command=self.point_featuring)
        self.process_btn.pack(pady=20)

    def create_right_Matches(self):
        ctk.CTkLabel(self.right_frame, text="Сопотавление точек").pack(pady=5)
        self.max_workers_slider, _ = self.create_slider_with_label(
            self.right_frame, "Количество ядер", 1, 4, initial=4, steps=4)
        self.good_distance_slider, _ = self.create_slider_with_label(
            self.right_frame, "good_distance", 0.1, 1, initial=0.75)
        self.ransacReprojThreshold_slider, _ = self.create_slider_with_label(
            self.right_frame, "ransacReprojThreshold", 0, 3, initial=1.5)
        self.maxIters_slider, _ = self.create_slider_with_label(
            self.right_frame, "maxIters", 2000, 10000, initial=2000, steps=10000-2000+1)
        self.process_btn = ctk.CTkButton(
            self.right_frame, text="Выделить", command=self.run_br_force_optimized_in_thread)
        self.process_btn.pack(pady=20)

    def create_right_Triangulate(self):
        ctk.CTkLabel(self.right_frame, text="Триангуляция точек").pack(pady=5)
        self.reprojectionError_slider, _ = self.create_slider_with_label(self.right_frame, "reprojectionError", 1, 10, initial=5)
        self.iterationsCount_slider, _ = self.create_slider_with_label(self.right_frame, "iterationsCount", 100, 5000, initial=1000)
        self.process_btn = ctk.CTkButton(self.right_frame, text="триангулировать", command=self.sfm)
        self.process_btn.pack(pady=20)

    def create_right_Filter(self):
        ctk.CTkLabel(self.right_frame, text="Фильтрация точек").pack(pady=5)
        self.min_x_slider, _ = self.create_slider_with_label(
            self.right_frame, "min_x", -50, 50, initial=0)
        self.max_x_slider, _ = self.create_slider_with_label(
            self.right_frame, "max_x", -50, 50, initial=0)
        self.min_y_slider, _ = self.create_slider_with_label(
            self.right_frame, "min_y", -50, 50, initial=0)
        self.max_y_slider, _ = self.create_slider_with_label(
            self.right_frame, "max_y", -50, 50, initial=0)
        self.min_z_slider, _ = self.create_slider_with_label(self.right_frame, "min_z", -50, 50, initial=0)
        self.max_z_slider, _ = self.create_slider_with_label(
            self.right_frame, "max_z", -50, 50, initial=0)
        ctk.CTkLabel(self.right_frame, text="Уплотнение точек").pack(pady=5)
        self.min_x_plot_slider, _ = self.create_slider_with_label(
            self.right_frame, "min_x", -50, 50, initial=0)
        self.max_x_plot_slider, _ = self.create_slider_with_label(
            self.right_frame, "max_x", -50, 50, initial=0)
        self.min_y_plot_slider, _ = self.create_slider_with_label(
            self.right_frame, "min_y", -50, 50, initial=0)
        self.max_y_plot_slider, _ = self.create_slider_with_label(
            self.right_frame, "max_y", -50, 50, initial=0)
        self.min_z_plot_slider, _ = self.create_slider_with_label(
            self.right_frame, "min_z", -50, 50, initial=0)
        self.max_z_plot_slider, _ = self.create_slider_with_label(
            self.right_frame, "max_z", -50, 50, initial=0)
        self.radius_dub_slider, _ = self.create_slider_with_label(
            self.right_frame, "Радиус точки", 0, 50, initial=0)
        self.points_dub_slider, _ = self.create_slider_with_label(
            self.right_frame, "Количество точек", 0, 50, initial=0)
        self.process_btn = ctk.CTkButton(
            self.right_frame, text="Фильтрация и уплотнение", command=self.dense_cloud)
        self.process_btn.pack(pady=20)

    def create_right_Reconstruct(self):
        ctk.CTkLabel(self.right_frame, text="Реконструкция поверхности Пуассона").pack(pady=5)
        self.dephth_tree_poisson_slider, _ = self.create_slider_with_label(self.right_frame, "dephth_tree", 1, 10, initial=8)
        self.densitythreshold_slider, _ = self.create_slider_with_label(self.right_frame, "densitythreshold", 0, 1, initial=0.001)

        self.normal_estimation_knn_slider, _ = self.create_slider_with_label(self.right_frame, "normal_estimation_knn", 1, 100, initial=25)
        self.normal_orientation_k_slider, _ = self.create_slider_with_label(self.right_frame, "normal_orientation_k", 1, 100, initial=33)
        self.remove_statistical_outliers_slider, _ = self.create_slider_with_label(self.right_frame, "remove_statistical_outliers", 0, 1, initial=0)
        self.nb_neighbors_slider, _ = self.create_slider_with_label(self.right_frame, "nb_neighbors", 1, 100, initial=1)
        self.std_ratio_slider, _ = self.create_slider_with_label(self.right_frame, "std_ratio", 1, 5, initial=1)
        self.mesh_simplify_slider, _ = self.create_slider_with_label(self.right_frame, "mesh_simplify", 0, 1, initial=0)
        self.target_number_of_triangles_slider, _ = self.create_slider_with_label(self.right_frame, "target_number_of_triangles", 100, 100000, initial=1000)
        self.process_btn = ctk.CTkButton(self.right_frame, text="Реконструкция", command=self.reconstruction)
        self.process_btn.pack(pady=20)

        self.process_btn = ctk.CTkButton(self.right_frame, text="Перевести в облако точек", command=self.meshtopointcloud)
        self.process_btn.pack(pady=20)










    def clear_log_file(self):
        log_file = "notes/debug_log.txt"
        try:
            if os.path.exists(log_file):
                with open(log_file, "w", encoding="utf-8") as f:
                    pass
        except Exception as e:
            self.log_message(f"Ошибка при очистке лог-файла: {e}")

    def clear_directories(self):
        directories = [
            'images/grayscales',
            'images/keypoints',
            'images/matches',
            'images/processed',
            'keypoints',
            'notes'
        ]

        for dir_path in directories:
            try:
                if os.path.exists(dir_path):
                    for filename in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f"Не удалось удалить {file_path}: {e}")

                    print(f"Директория {dir_path} очищена")
                else:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Директория {dir_path} создана")
                if dir_path == 'notes':
                    debug_log_path = os.path.join(dir_path, 'debug_log.txt')
                    with open(debug_log_path, 'w') as f:
                        f.write('Debug log created\n')
                    print(f"Файл {debug_log_path} создан")
            except Exception as e:
                print(f"Ошибка при очистке {dir_path}: {e}")

    def __init__(self):
        super().__init__()
        self.title("3D Reconstructor")
        self.geometry(
            f"{1000}x{600}+{(self.winfo_screenwidth() // 2) - (1000 // 2)}+{(self.winfo_screenheight() // 2) - (600 // 2)}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log_message = log_message.__get__(self)
        self.clear_log_file()
        self.clear_directories()

        self.images = ImageData()
        self.features = FeatureData()
        self.current_image = None
        self.error = 0
        self.steps = {
            'load':False,
            'processed': False,
            'grayscale': False,
            'featuring': False,
            'bf_match': False,
            'sfm': False,
            }

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=6)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_left_panel()
        self.create_central_panel()
        self.create_right_panel()

    def on_closing(self):
        if hasattr(self, 'after_ids'):
            for id_ in list(self.after_ids):
                self.after_cancel(id_)
            del self.after_ids
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
        self.destroy()

    def update_slider_label(self, value, slider=None, label=None):
        if label:
            label.configure(text=f"{value:.2f}")

    def create_slider_with_label(self, parent, label_text, from_, to, initial=0, steps=None, precision=2):
        group_frame = ctk.CTkFrame(parent)
        group_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(group_frame, text=label_text).pack(anchor="w")

        slider_frame = ctk.CTkFrame(group_frame)
        slider_frame.pack(fill="x", pady=(0, 5))

        slider = ctk.CTkSlider(
            slider_frame,
            from_=from_,
            to=to
        )
        if steps:
            slider.configure(number_of_steps=steps)
        slider.pack(side="left", expand=True, fill="x", padx=(0, 10))
        slider.set(initial)

        value_label = ctk.CTkLabel(slider_frame, text=f"{initial:.{precision}f}", width=50)
        value_label.pack(side="left")

        entry = ctk.CTkEntry(
            slider_frame,
            width=70,
            validate="key",
            validatecommand=(parent.register(self.validate_float_input), "%P"
        ))
        entry.insert(0, f"{initial:.{precision}f}")
        self.setup_slider_connections(slider, value_label, entry, precision)
        entry.pack(side="left", padx=(10, 0))
        return slider, value_label

    def setup_slider_connections(self, slider, label, entry, precision):
        self._updating = False

        def on_slider_change(value):
            if not self._updating:
                self._updating = True
                formatted_value = f"{float(value):.{precision}f}"
                label.configure(text=formatted_value)
                entry.delete(0, "end")
                entry.insert(0, formatted_value)
                self._updating = False

        def on_entry_change(event=None):
            if not self._updating:
                try:
                    self.log_message(f"Начало обработки ввода. Текущее значение поля: {entry.get()}")
                    value = float(entry.get())
                    self.log_message(f"Преобразованное значение: {value}")

                    min_val = slider.cget("from_")
                    max_val = slider.cget("to")
                    self.log_message(f"Допустимый диапазон: от {min_val} до {max_val}")

                    value = max(min_val, min(max_val, value))
                    self.log_message(f"Ограниченное значение: {value}")

                    self._updating = True
                    self.log_message("Устанавливаем флаг _updating")

                    slider.set(value)
                    self.log_message("Значение слайдера установлено")

                    formatted_value = f"{value:.{precision}f}"
                    self.log_message(f"Форматированное значение: {formatted_value}")

                    label.configure(text=formatted_value)
                    self.log_message("Метка обновлена")

                    entry.delete(0, "end")
                    entry.insert(0, formatted_value)
                    self.log_message("Поле ввода обновлено")

                    self._updating = False
                    self.log_message("Сброшен флаг _updating")

                except Exception as e:
                    self.log_message(f"Произошла ошибка: {str(e)}")
                    current_value = float(label.cget("text"))
                    self.log_message(f"Восстанавливаем предыдущее значение: {current_value}")
                    entry.delete(0, "end")
                    entry.insert(0, f"{current_value:.{precision}f}")
                    self._updating = False
                    self.log_message("Сброшен флаг _updating после ошибки")
        slider.configure(command=on_slider_change)
        entry.bind("<Return>", on_entry_change)
        entry.bind("<FocusOut>", on_entry_change)

    def validate_float_input(self, new_value):
        try:
            if new_value.strip() == "":
                return True
            float(new_value)
            return True
        except ValueError:
            return False














# Загрузка изображений и проверка
    def load_images(self):
        try:
            self.log_message("Загрузка изображений")
            files = filedialog.askopenfilenames(
                initialdir=os.path.expanduser("./"),
                filetypes=[
                    ("Все изображения", "*.jpeg *.jpe *.png *.jpg *.webp"),
                    ("JPEG/JPG", "*.jpeg *.jpg"),
                    ("PNG", "*.png")
                ]
            )
            existing_paths = {os.path.normpath(p) for p in self.images.paths}
            new_files = []
            for f in files:
                norm_path = os.path.normpath(f)
                if norm_path not in existing_paths:
                    new_files.append(f)
                    existing_paths.add(norm_path)
            if new_files:
                self.images.paths.extend(new_files)
                self.update_image_list()
                self.log_message(f"Изображения загруженны: {self.images.paths}")
                self.error = 0
                self.steps = {
                    'load': True,
                    'processed': False,
                    'grayscale': False,
                    'featuring': False,
                    'bf_match': False,
                    'sfm': False,
                    }
                self.start_validation()
        except Exception as e:
            print(e)
            self.error = 4
            error_handle(self.error)

    def start_validation(self):
        if not self.images.paths:
            self.error = 1
            error_handle(self.error)
            return
        if len(self.images.paths) < 2:
            self.error = 2
            error_handle(self.error)
            return
        self.error = 0
        threading.Thread(target=self.validate_images, daemon=True).start()

    def validate_images(self):
        try:
            progress_window = ProgressWindow(self, len(self.images.paths))
            results = {
                'count_ok': check_image_count(self.images.paths),
                'empty_paths': check_empty_paths(self.images.paths),
                'images': []
            }
            if not results['empty_paths']:
                for i, path in enumerate(self.images.paths, 1):
                    try:
                        progress_window.update_progress(i, f"Проверка: {path}")
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
                        time.sleep(0.5)
                        self.error = 0
                    except Exception as e:
                        self.error = 3
                        error_handle(self.error, f"Ошибка при проверке {path}: {e}")
            progress_window.destroy()
            self.after(0, lambda: ResultsWindow(self, results))
        except:
            self.error = 5
            error_handle(self.error)

    def update_image_list(self):
        for widget in self.image_list_frame.winfo_children():
            widget.destroy()
        for path in self.images.paths:
            frame = ctk.CTkFrame(self.image_list_frame)
            frame.pack(fill="x", pady=2)
            img = Image.open(path)
            img.thumbnail((50, 50))
            photo = ImageTk.PhotoImage(img)
            label = ctk.CTkLabel(frame, image=photo, text="")
            label.image = photo
            label.pack(side="left")
            path_label = ctk.CTkLabel(frame, text=os.path.basename(path))
            path_label.pack(side="left", padx=10)
            remove_btn = ctk.CTkButton(frame, text="X", width=30, command=lambda p=path: self.remove_image(p))
            remove_btn.pack(side="right")

    def remove_image(self, path):
        self.images.paths.remove(path)
        self.update_image_list()
        self.log_message(f"Изображение удалено: {path}")

# Улучшение изображений
    def process_images(self):
        print('\n\n\n\n\n\n\n\nprocess_images', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        print
        try:
            if self.error not in [1,2]:
                if self.steps['load'] == False:
                    self.error = 1
                    error_handle(self.error)
                else:
                    contrast = self.contrast_slider.get()
                    bright = self.bright_slider.get()
                    sharp_radius = int(self.radius_slider.get())
                    sharp_percent = int(self.percent_slider.get())
                    sharp_threshold = int(self.threshold_slider.get())

                    color_balance = (self.color_balance_x_slider.get(), self.color_balance_y_slider.get(), self.color_balance_z_slider.get())
                    hist_normalize = True if self.hist_normalize_slider.get()=="on" else False
                    denoise_strength = self.denoise_strength_slider.get()
                    denoise_template = int(self.denoise_template_slider.get())
                    denoise_search = int(self.denoise_search_slider.get())

                    self.log_message("Обработка изображений...")
                    self.log_message(f"Параметры: contrast={contrast}\nbright={bright}\nradius={sharp_radius}\npercent={sharp_percent}\nthreshold={sharp_threshold}")

                    self.images.processed = Process_Images(self.images.paths, bright, contrast, color_balance, hist_normalize, denoise_strength, denoise_template, denoise_search, sharp_radius, sharp_percent, sharp_threshold, log_func=self.log_message)
                    self.steps['processed'] = True
                    self.error = 0
                    # if self.images.processed:
                        # try:
                            # self.show_visualisation(self.images.processed[0])
                    self.log_message("Все изображения обработаны")

                        # except Exception as e:
                            # self.error=7
                            # error_handle(self.error)
                            # self.visualization_label.configure(text="Ошибка загрузки изображения")
            else:
                error_handle(self.error)
        except:
            self.error = 6
            error_handle(self.error)

    def grayscale_images(self):
        print('\n\n\n\n\n\n\n\ngrayscale_images', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 6,7,8,9,10, 11, 12]:
            try:
                if self.steps['processed'] == False:
                    self.error = 14
                    error_handle(self.error)
                    self.images.processed = self.images.paths

                minadge = self.minedge_slider.get()
                maxedge = self.maxedge_slider.get()
                kernel_size = int(self.kernel_size_slider.get())
                clahe_clip_limit = self.clahe_clip_limit_slider.get()
                clahe_grid_size = int(self.clahe_grid_size_slider.get())
                noise_checker = self.noise_checker.get()
                noise_mean = self.noise_mean_slider.get()
                noise_stddev = self.noise_stddev_slider.get()
                blur_kernel_size = int(self.blur_kernel_size_slider.get())
                blur_sigma = self.blur_sigma_slider.get()
                self.log_message("Обработка серых изображений...")
                self.log_message(f"Параметры: minadge={minadge}\nmaxedge={maxedge}\nkernel_size={kernel_size}\nclahe_clip_limit={clahe_clip_limit}\nclahe_grid_size={clahe_grid_size}\nnoise_mean={noise_mean}\nnoise_stddev={noise_stddev}\nblur_kernel_size={blur_kernel_size}\nblur_sigma={blur_sigma}\n\n")
                self.images.grayscale, self.images.masks = GrayScale_Images(
                    self.images.processed, noise_checker, minadge, maxedge, kernel_size, clahe_clip_limit, clahe_grid_size, noise_mean, noise_stddev, blur_kernel_size, blur_sigma, log_func=self.log_message)
                self.log_message("Все изображения обработаны")
                self.steps['grayscale'] = True
            except:
                self.error = 7
                error_handle(self.error)
        else:
            error_handle(self.error)

# Выделение точек
    def point_featuring(self):
        print('\n\n\n\n\n\n\n\npoint_featuring', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 8,  9, 10, 11, 12]:
            try:
                if self.steps['grayscale'] == False:
                    self.error = 15
                    error_handle(self.error)
                else:
                    sift_point = int(self.SIFT_slider.get())
                    random_point = int(self.Random_slider.get())
                    nOctaveLayers = int(self.nOctaveLayers_slider.get())
                    contrastThreshold = self.contrastThreshold_slider.get()
                    edgeThreshold = int(self.edgeThreshold_slider.get())
                    sigma = self.sigma_slider.get()
                    self.log_message("Выделение точек...")
                    self.log_message(f'sift_point {sift_point}\nrandom_point {random_point}\nnOctaveLayers {nOctaveLayers}\ncontrastThreshold {contrastThreshold}\nedgeThreshold {edgeThreshold}\nsigma {sigma}')
                    self.features.keypoints, self.features.descriptors = Point_Featuring(self.images.grayscale, self.images.masks, sift_point, random_point,
                                                        nOctaveLayers, contrastThreshold, edgeThreshold, sigma, log_func=self.log_message)
                    self.log_message("Все изображения обработаны")
                    self.steps['featuring'] = True
                    self.error = 0
            except:
                self.error=8
                error_handle(self.error)
        else:
            error_handle(self.error)
# Сопоставление точек
    def run_br_force_optimized_in_thread(self):
        def task():
            self.br_force_optimized()
            self.after(0, self.after_br_force_optimized)
        Thread(target=task).start()
    def after_br_force_optimized(self):
        write_ply_from_matches(self.features.matches, self.features.keypoints,
                               self.images.processed, log_func=self.log_message)
        self.show_visualisation("keypoints/matches.ply")
        self.log_message("Сопоставление завершено")
    def br_force_optimized(self):
        print('\n\n\n\n\n\n\n\nbr_force_optimized', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 9, 10, 11, 12]:
            try:
                if self.steps['featuring'] == False:
                    self.error = 16
                    error_handle(self.error)
                else:
                    max_workers = int(self.max_workers_slider.get())
                    good_distance = (self.good_distance_slider.get())
                    ransacReprojThreshold = (self.ransacReprojThreshold_slider.get())
                    maxIters = int(self.maxIters_slider.get())
                    self.log_message("Сопоставление точек...")
                    self.log_message(f'max_workers {max_workers}\ngood_distance {good_distance}\nransacReprojThreshold {ransacReprojThreshold}\nmaxIters {maxIters}')
                    self.features.matches = br_force_optimized(self.features.keypoints, self.features.descriptors, self.images.grayscale, visualize=True, max_workers=max_workers, good_distance=good_distance, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, log_func=self.log_message)
                    self.steps['bf_match'] = True
                    self.log_message("Сопоставление закончено")
                    self.error = 0
            except:
                self.error = 9
                error_handle(self.error)
        else:
            error_handle(self.error)


# 3D-облако
    def sfm(self):
        print('\n\n\n\n\n\n\n\nsfm', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 10, 11, 12]:
            try:
                if self.steps['bf_match'] == False:
                    self.error = 17
                    error_handle(self.error)
                else:
                    h, w = cv2.imread(self.images.paths[0]).shape[:2]
                    focal = 0.8 * max(h, w)
                    K = np.array([[focal, 0, w / 2],
                                [0, focal, h / 2],
                                [0, 0, 1]], dtype=np.float64)
                    reprojectionError = (self.reprojectionError_slider.get())
                    iterationsCount = int(self.iterationsCount_slider.get())
                    self.log_message("Востановление положений камер...")
                    self.log_message(f'height {h}\nwidth {w}\nfocal {focal}\nматрица камеры {K}\nreprojectionError {reprojectionError}\niterationsCount {iterationsCount}')
                    points3d, camera_poses, point3d_ids = run_sfm_pipeline(
                        K, self.features.keypoints, self.features.matches, log_func=self.log_message)
                    colors = colorize_points(
                        points3d, point3d_ids, self.features.keypoints, self.images.paths)
                    save_point_cloud_ply_with_color("keypoints/point_cloud_colored.ply", points3d, colors)
                    self.features.point_cloud['points'] = points3d
                    self.features.point_cloud['colors'] = colors
                    self.log_message("Триангуляция завершена")
                    self.steps['sfm'] = True
                    self.error = 0
            except:
                self.error = 10
                error_handle(self.error)
        else:
            error_handle(self.error)

# Уплотнение
    def dense_cloud(self):
        print('\n\n\n\n\n\n\n\ndense_cloud', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 11, 12]:
            try:
                x = [self.min_x_slider.get(), self.max_x_slider.get()]
                y = [self.min_y_slider.get(), self.max_y_slider.get()]
                z = [self.min_z_slider.get(), self.max_z_slider.get()]
                self.features.point_cloud['points'], self.features.point_cloud['colors'] = dense_point_cloud(
                    self.features.point_cloud['points'], self.features.point_cloud['colors'], x=x, y=y, z=z, log_func=self.log_message)
                x_plot_ = [self.min_x_plot_slider.get(), self.max_x_plot_slider.get()]
                y_plot_ = [self.min_y_plot_slider.get(), self.max_y_plot_slider.get()]
                z_plot_ = [self.min_z_plot_slider.get(), self.max_z_plot_slider.get()]
                radius_dub = self.radius_dub_slider.get()
                points_dub = int(self.points_dub_slider.get())
                if (x_plot_[0] == 0 and x_plot_[1] == 0 and y_plot_[0] == 0 and y_plot_[1] == 0 and z_plot_[0] == 0 and z_plot_[1] == 0):
                    save_ply("keypoints/dense_point_cloud.ply", self.features.point_cloud['points'],self.features.point_cloud['colors'], color_order="bgr")
                    self.show_visualisation("keypoints/dense_point_cloud.ply")
                if radius_dub!=0:
                    dense_pcd = duplicate_around_points(self.features.point_cloud['points'], self.features.point_cloud['colors'], radius=radius_dub, points_per_point=points_dub,
                                                    x_min=x_plot_[0], x_max=x_plot_[1],
                                                    y_min=y_plot_[0], y_max=y_plot_[1],
                                                    z_min=z_plot_[0], z_max=z_plot_[1],
                                                    )
                    o3d.io.write_point_cloud("keypoints/dense_point_cloud.ply", dense_pcd) #, self.features.point_cloud['points'], self.features.point_cloud['colors']
                else:
                    save_point_cloud_ply_with_color("keypoints/dense_point_cloud.ply", self.features.point_cloud['points'], self.features.point_cloud['colors'])
                self.show_visualisation("keypoints/dense_point_cloud.ply")
                self.log_message("Фильтрация и уплотнение завершено")
                print('\n\n\n\n\n\n\n\ndense_cloud', self.error, self.steps, '\n\n\n\n\n\n\n\n')
                self.error = 0
            except Exception as e:
                self.error=11
                error_handle(self.error)
                print('\n\n\n\n\n\n\n\ndense_cloud!!!!!!!!!!!!!!!!!!!except\ne',e, self.error, self.steps, '\n\n\n\n\n\n\n\n')
        else:
            error_handle(self.error)

# Построение модели
    def reconstruction(self):
        print('\n\n\n\n\n\n\n\nreconstruction', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 12]:
            try:
                dephth_tree_poisson_slider = int(self.dephth_tree_poisson_slider.get())
                densitythreshold_slider = (self.densitythreshold_slider.get())
                normal_estimation_knn = int(self.normal_estimation_knn_slider.get())
                normal_orientation_k = int(self.normal_orientation_k_slider.get())
                remove_statistical_outliers = True if int(self.remove_statistical_outliers_slider.get())==1 else False
                nb_neighbors = int(self.nb_neighbors_slider.get())
                std_ratio = (self.std_ratio_slider.get())
                mesh_simplify = True if int(self.mesh_simplify_slider.get())==1 else False,
                target_number_of_triangles = int(self.target_number_of_triangles_slider.get())
                points = self.features.point_cloud['points']
                colors = self.features.point_cloud['colors']
                pcd = o3d.geometry.PointCloud()
                if not isinstance(points, np.ndarray):
                    points = np.array(points, dtype=np.float32)
                if points.shape[1] != 3:
                    raise ValueError("Points must have shape (N, 3)")
                pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    if not isinstance(colors, np.ndarray):
                        colors = np.array(colors, dtype=np.float32)
                    if colors.shape[1] != 3:
                        raise ValueError("Colors must have shape (N, 3)")
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                self.features.mesh, stats = reconstruct_surface_from_point_cloud(
                    pcd,
                    poisson_depth=dephth_tree_poisson_slider,
                    density_threshold_percentile=densitythreshold_slider,
                    normal_estimation_knn=normal_estimation_knn,
                    normal_orientation_k=normal_orientation_k,
                    remove_statistical_outliers=remove_statistical_outliers,
                    nb_neighbors=nb_neighbors,
                    std_ratio=std_ratio,
                    mesh_simplify=mesh_simplify,
                    target_number_of_triangles=target_number_of_triangles
                )
                self.log_message("Реконструкция завершена")
                o3d.io.write_triangle_mesh("keypoints/mesh_poisson.ply", self.features.mesh)
                self.show_visualisation("keypoints/mesh_poisson.ply")
                self.error = 0
            except:
                self.error=12
                error_handle(self.error)
        else:
            error_handle(self.error)

    def meshtopointcloud(self):
        print('\n\n\n\n\n\n\n\n', self.error, self.steps, '\n\n\n\n\n\n\n\n')
        if self.error in [0, 13]:
            try:
                self.features.point_cloud['points'], self.features.point_cloud['colors'] = from_mesh_to_pointcloud(self.features.mesh)
                self.log_message("Получено облако точек")
                save_point_cloud_ply_with_color("keypoints/meshtopointcloud_point_cloud.ply",
                                                self.features.point_cloud['points'],
                                                self.features.point_cloud['colors'])
                self.error = 0
            except:
                self.error=13
                error_handle(self.error)
        else:
            error_handle(self.error)

















    def clear_visualisation(self):
        if hasattr(self, 'visualization_label') and self.visualization_label is not None:
            self.visualization_label.configure(image=None, text="")
            self.visualization_label.image = None

    def show_visualisation(self, path):
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
        if path.lower().endswith('.ply'):
            self.log_message("отображение ply...")
            self.show_ply_in_frame(path)
        else:
            self.log_message("отображение path...")
            self.show_image_in_frame(path)

    def show_image_in_frame(self, image_path):
        image = Image.open(image_path)

        photo = ImageTk.PhotoImage(image)
        label = ctk.CTkLabel(self.visualization_frame, text=image_path, image=photo)
        label.image = photo
        label.pack(expand=True, fill="both")














    def show_ply_in_frame(self, ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b')
        ax.set_title(ply_path)
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)





































if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

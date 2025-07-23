from typing import Tuple, Optional
import numpy as np
import open3d as o3d


def read_ply_ascii_with_colors(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_vertices = None
    end_header_idx = -1
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        if line_clean.startswith("element vertex"):
            try:
                num_vertices = int(line_clean.split()[-1])
            except ValueError:
                raise ValueError(
                    f"Не удалось извлечь количество вершин из строки: {line}")
        if line_clean == "end_header":
            end_header_idx = i
            break
    if end_header_idx == -1:
        raise ValueError("Файл не содержит 'end_header'")
    if num_vertices is None:
        raise ValueError(
            "Файл не содержит 'element vertex' для определения количества вершин")
    data_lines = lines[end_header_idx + 1:end_header_idx + 1 + num_vertices]
    points = np.zeros((num_vertices, 3), dtype=np.float32)
    colors = np.zeros((num_vertices, 3), dtype=np.uint8)
    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        if len(parts) < 6:
            raise ValueError(f"Недостаточно данных в строке {i}: {line}")
        points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        colors[i] = [int(parts[3]), int(parts[4]), int(parts[5])]

    return points, colors


def reconstruct_surface_from_point_cloud(
    pcd: o3d.geometry.PointCloud,
    poisson_depth: int = 8,
    density_threshold_percentile: float = 0.01,
    normal_estimation_knn: int = 30,
    normal_orientation_k: int = 100,
    remove_statistical_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    mesh_simplify: bool = True,
    target_number_of_triangles: int = 100000
) -> Tuple[Optional[o3d.geometry.TriangleMesh], dict]:

    stats = {
        'input_points': len(pcd.points),
        'parameters': {
            'poisson_depth': poisson_depth,
            'density_threshold_percentile': density_threshold_percentile,
            'normal_estimation_knn': normal_estimation_knn,
            'normal_orientation_k': normal_orientation_k
        }
    }

    try:

        if remove_statistical_outliers:
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            pcd = pcd.select_by_index(ind)
            stats['after_outlier_removal_points'] = len(pcd.points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(
                knn=normal_estimation_knn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(normal_orientation_k)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=poisson_depth
        )
        stats['initial_mesh_vertices'] = len(mesh.vertices)
        stats['initial_mesh_triangles'] = len(mesh.triangles)

        densities = np.asarray(densities)
        density_threshold = np.quantile(
            densities, density_threshold_percentile)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        stats['after_density_filter_vertices'] = len(mesh.vertices)
        stats['density_threshold'] = float(density_threshold)

        if mesh_simplify and len(mesh.triangles) > target_number_of_triangles:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
            stats['simplified_mesh_triangles'] = len(mesh.triangles)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        stats['final_vertices'] = len(mesh.vertices)
        stats['final_triangles'] = len(mesh.triangles)
        mesh.compute_vertex_normals()
        return mesh, stats

    except Exception as e:
        print(f"Ошибка реконструкции: {str(e)}")
        stats['error'] = str(e)
        return None, stats


def from_mesh_to_pointcloud(mesh):
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise ValueError("Input must be an Open3D TriangleMesh")

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    if mesh.has_vertex_colors():

        print("colors exsist")
        pcd.colors = mesh.vertex_colors
    elif mesh.has_triangle_colors():

        print("triangles exist")
        triangle_colors = np.asarray(mesh.triangle_colors)
        vertex_colors = np.zeros((len(mesh.vertices), 3))

        for i, vertex in enumerate(mesh.vertices):
            triangles = [ti for ti, tri in enumerate(
                mesh.triangles) if i in tri]
            if triangles:
                vertex_colors[i] = triangle_colors[triangles].mean(axis=0)
        pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        print("colors doesnt exsist")

        pcd.paint_uniform_color((255, 255, 255))

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)

    return points, colors

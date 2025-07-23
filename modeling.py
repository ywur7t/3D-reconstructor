import numpy as np
import open3d as o3d

def pointcloud_to_mesh(ply_file):

    pcd = o3d.io.read_point_cloud(ply_file)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)





def pointcloud_to_mesh_alpha_shape(ply_file, alpha=5000):

    pcd = o3d.io.read_point_cloud(ply_file)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("output_mesh_alpha.ply", mesh)




def extract_planes_and_build_mesh(ply_file, plane_threshold=0.03, min_points=100):
    # Загружаем облако точек
    pcd = o3d.io.read_point_cloud(ply_file)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    planes = []
    remaining = pcd
    all_meshes = []

    while True:
        # Плоскость через RANSAC
        plane_model, inliers = remaining.segment_plane(distance_threshold=plane_threshold,
                                                       ransac_n=3,
                                                       num_iterations=10000)
        if len(inliers) < min_points:
            break

        # Извлекаем плоскость
        plane_cloud = remaining.select_by_index(inliers)
        hull, _ = plane_cloud.compute_convex_hull()
        hull.compute_vertex_normals()
        all_meshes.append(hull)

        # Удаляем найденные точки
        remaining = remaining.select_by_index(inliers, invert=True)

    if not all_meshes:
        print("Не найдено плоскостей.")
        return

    # Объединяем все mesh'и
    full_mesh = all_meshes[0]
    for m in all_meshes[1:]:
        full_mesh += m

    o3d.visualization.draw_geometries([full_mesh], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("output_planes_mesh.ply", full_mesh)




def Simplification(ply_file):

    # Загружаем mesh
    mesh = o3d.io.read_triangle_mesh(ply_file)

    # Упрощаем модель (число треугольников = 5000)
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

    # Обновляем нормали
    simplified.compute_vertex_normals()

    # Сохраняем и визуализируем
    # o3d.visualization.draw_geometries([simplified])
    o3d.io.write_triangle_mesh("optimized_mesh.ply", simplified)


def Cleaning(ply_file):

    mesh = o3d.io.read_triangle_mesh(ply_file)
    # Удалить висячие вершины
    mesh.remove_unreferenced_vertices()

    # Удалить малые компоненты (например, мусор от сканов)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("optimized_mesh.ply", mesh)



def optimize_and_export_obj(input_ply, output_obj, target_triangles=3000):
    # Загрузка mesh
    mesh = o3d.io.read_triangle_mesh(input_ply)

    # Очистка
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    # Упрощение
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # Экспорт в .obj
    o3d.io.write_triangle_mesh(output_obj, mesh, write_triangle_uvs=False)

    print(f"[✓] Упрощённый .obj сохранён как {output_obj}")
    return mesh




Simplification("output_mesh_alpha.ply")
# Cleaning("optimized_mesh.ply")
optimize_and_export_obj("optimized_mesh.ply", "light_model.obj")
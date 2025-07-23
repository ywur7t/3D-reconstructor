def save_keypoints_as_pointcloud(points_with_depth, filename='keypoints_cloud.ply'):
    """
    Сохраняет ключевые точки с глубиной как облако точек в формате PLY.

    **Входные параметры**:
    points_with_depth – список (x, y, z)
    filename – имя выходного файла (.ply)

    **Выходные параметры**:
    **нет**
    """
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points_with_depth)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for x, y, z in points_with_depth:
            # if z < 1000:
            f.write(f'{x} {y} {z}\n')


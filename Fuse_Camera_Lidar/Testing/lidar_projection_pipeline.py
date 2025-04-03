
import numpy as np
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_velodyne_bin(path):
    points = []
    with open(path, "rb") as f:
        while True:
            bytes_read = f.read(16)
            if not bytes_read:
                break
            x, y, z, _ = struct.unpack("ffff", bytes_read)
            points.append([x, y, z])
    return np.array(points)

def parse_matrix_line(line, shape):
    return np.array([float(x) for x in line.split()[1:]]).reshape(shape)

def load_calib_velo_to_cam(path):
    with open(path) as f:
        lines = f.readlines()
    R = parse_matrix_line(lines[1], (3, 3))
    T = np.array([float(x) for x in lines[2].split()[1:]]).reshape((3, 1))
    return R, T

def load_calib_cam_to_cam(path):
    with open(path) as f:
        lines = f.readlines()
    R_rect = parse_matrix_line(
        [l for l in lines if l.startswith("R_rect_02:")][0], (3, 3)
    )
    P_rect = parse_matrix_line(
        [l for l in lines if l.startswith("P_rect_02:")][0], (3, 4)
    )
    return R_rect, P_rect

def project_lidar_points_to_image(lidar_points, R, T, R_rect, P_rect, image):
    image_with_points = image.copy()
    projected_uvz = []
    for point in lidar_points:
        point = point.reshape(3, 1)
        cam_coords = R @ point + T
        if cam_coords[2] <= 0:
            continue
        rectified = R_rect @ cam_coords
        rectified_hom = np.vstack((rectified, [[1]]))
        uvw = P_rect @ rectified_hom
        u = int(uvw[0] / uvw[2])
        v = int(uvw[1] / uvw[2])
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            depth = rectified[2, 0]
            projected_uvz.append((u, v, depth))
            norm_depth = np.clip(depth / 50, 0, 1)
            color = tuple([int(c * 255) for c in cm.jet(1 - norm_depth)[:3]])
            cv2.circle(image_with_points, (u, v), 1, color, -1)
    return image_with_points, projected_uvz

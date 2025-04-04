import laspy
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import argparse
import cv2
import json
import re
import os
from scipy.spatial.transform import Rotation


def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T


def relative_transform(pose_a, pose_b):
    """Compute relative rotation R and translation t between two 7D poses."""
    T_a = pose_to_matrix(*pose_a)
    T_b = pose_to_matrix(*pose_b)

    T_ab = np.linalg.inv(T_a) @ T_b
    R = T_ab[:3, :3]
    t = T_ab[:3, 3]

    return R, t


def enumerate_cam_images(folder_path):
    pattern = re.compile(r'cam_.*_(\d+)\.jpg')
    files = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            timestamp = int(match.group(1))
            files.append((timestamp, folder_path + "/" + filename))
    files.sort(key=lambda x: x[0])
    return files


class ImageBuffer:
    def __init__(self, all_images):
        self.images = all_images

    def get_images_up_to(self, timestamp):
        result = [img for img in self.images if img[0] <= timestamp]
        self.images = [img for img in self.images if img[0] > timestamp]
        return result


def get_dataset_image_buffer(traj_data, dataset_folder, subsample=1):
    all_images = []
    for cam_name in traj_data["camera_infos"]:
        cam_path = dataset_folder + "/" + cam_name
        cam_images = enumerate_cam_images(cam_path)
        ii = 0
        for timestamp, cam_frame_path in cam_images:
            ii = ii + 1
            if ii % subsample != 0:
                continue
            all_images.append((timestamp, cam_name, cam_frame_path))
    all_images.sort(key=lambda x: x[0])
    return ImageBuffer(all_images)


def visualize_laz(file_path: str, traj_data, image_buffer, send_laser_intensities):
    index = json.load(open(file_path + ".index.json"))["entries"]
    index.sort(key = lambda e : e["end_ts"])
    if any([e["part_index"] != 0 for e in index]):
        raise RuntimeError("TODO: multipart laz export not supported for now.")

    with laspy.open(file_path) as las:
        print(f"Number of points in dataset: {las.header.point_count}")
        # print("Available point attributes:")
        # for dimension in las.header.point_format.dimensions:
        #     print(f"- {dimension.name}")

        # Reset file pointer for full point cloud processing
        las.seek(0)

        for entry in index:
            las.seek(entry["point_offset_in_file"])
            points = las.read_points(entry["num_pts"])
            coords = np.vstack((points.x, points.y, points.z)).T

            colors = None
            # if hasattr(points, 'red') and hasattr(points, 'green') and hasattr(points, 'blue'):
            #     colors = np.vstack((points.red, points.green, points.blue)).T / 65535.0  # Normalize to [0,1]
            if send_laser_intensities and hasattr(points, 'intensity'):
                colors = np.vstack((points.intensity, points.intensity, points.intensity)).T / 65535.0  # Normalize to [0,1]

            end_ts = entry["end_ts"]

            images = image_buffer.get_images_up_to(end_ts)
            for timestamp, cam_name, cam_frame_path in images:
                camera_info = traj_data['camera_infos'][cam_name];
                rr_send_cam_frame(timestamp + camera_info['timestamp_offset'], cam_name, camera_info, cam_frame_path);

            rr.set_time_seconds("sensor_time", end_ts / (1000.0 * 1000.0))

            # radius: positive = scene units, negative = ui points
            rr.log("world/" + entry["laser_name"], rr.Points3D(coords, colors=colors, radii=-1))

def rr_send_sensor_extrinsics(traj_data):
    rr.log("world/rig/axis", rr.Arrows3D(
        vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
    for frame_name in traj_data["sensor_to_trajectory_poses"]:
        pose = traj_data["sensor_to_trajectory_poses"][frame_name]
        rr.log(f"world/rig/{frame_name}", rr.Transform3D(translation=[pose[0], pose[1], pose[2]],
                                                         rotation=rr.Quaternion(
                                                             xyzw=[pose[4], pose[5], pose[6], pose[3]])), static=True)
        rr.log(f"world/rig/{frame_name}", rr.Arrows3D(
            vectors=[[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)


def rr_send_rig_trajectory(traj_data):
    timestamps = []
    positions = []
    quaternions_xyzw = []
    for timestamp, pose in traj_data["trajectory_poses"]:
        x, y, z, qw, qx, qy, qz = pose
        timestamps.append(timestamp / (1000.0 * 1000.0))
        positions.append([x, y, z])
        quaternions_xyzw.append([qx, qy, qz, qw])

        # rr.set_time_seconds("sensor_time", timestamp/(1000.0*1000.0))
        # rr.log("world/rig", rr.Transform3D(translation=[x, y, z], quaternion=[qx, qy, qz, qw]))

    rr.send_columns("world/rig", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                    columns=rr.Transform3D.columns(translation=positions, quaternion=quaternions_xyzw))
    # permanently plot a line along the trajectory
    rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)


def rr_send_cam_frame(timestamp, cam_name, camera_info, cam_frame_path):
    image = cv2.imread(cam_frame_path)

    # Ensure the image was loaded correctly
    if image is None:
        raise ValueError(f"Image '{cam_frame_path}' could not be loaded. Make sure the file path is correct.")

    fx, fy = camera_info['fx'], camera_info['fy']
    cx, cy = camera_info['cx'], camera_info['cy']

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

    dist_coeffs = np.array(camera_info['dist_coeffs'])
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    rr.set_time_seconds("sensor_time", timestamp / (1000 * 1000.0))

    rr.log(f"world/rig/{cam_name}/undistorted_image",
           rr.Pinhole(focal_length=[fx, fy], principal_point=[cx, cy], width=camera_info["image_width"],
                      height=camera_info["image_height"], image_plane_distance=0.1))
    rr.log(f"world/rig/{cam_name}/undistorted_image",
           rr.Image(undistorted_image, color_model="BGR").compress(jpeg_quality=75))


def rr_send_imu_data(sensor_export):
    if "imu" in sensor_export:
        imu_data = sensor_export["imu"]
        timestamps = np.array([entry["timestamp"] / (1000.0 * 1000.0) for entry in imu_data])
        accel_x = np.array([entry["accel_x"] for entry in imu_data])
        accel_y = np.array([entry["accel_y"] for entry in imu_data])
        accel_z = np.array([entry["accel_z"] for entry in imu_data])
        rr.send_columns("imu/accel/x", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                        columns=rr.Scalar.columns(scalar=accel_x))
        rr.send_columns("imu/accel/y", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                        columns=rr.Scalar.columns(scalar=accel_y))
        rr.send_columns("imu/accel/z", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                        columns=rr.Scalar.columns(scalar=accel_z))

        gyro_x = np.array([entry["gyro_x"] for entry in imu_data])
        gyro_y = np.array([entry["gyro_y"] for entry in imu_data])
        gyro_z = np.array([entry["gyro_z"] for entry in imu_data])
        rr.send_columns("imu/gyro/x", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                        columns=rr.Scalar.columns(scalar=gyro_x))
        rr.send_columns("imu/gyro/y", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                        columns=rr.Scalar.columns(scalar=gyro_y))
        rr.send_columns("imu/gyro/z", indexes=[rr.TimeSecondsColumn("sensor_time", timestamps)],
                        columns=rr.Scalar.columns(scalar=gyro_z))


def rr_send_gps_data(sensor_export):
    if "gps" in sensor_export:
        gps_data = sensor_export["gps"]
        
        gps_traj = [[g['lat_rad']/ np.pi * 180.0, g['lon_rad']/ np.pi * 180.0] for g in gps_data]
        rr.log("gps/trajectory",rr.GeoLineStrings(lat_lon=gps_traj,
                                                  radii=rr.Radius.ui_points(2.0),
                                                  colors=[0, 0, 255]), static=True)
        for g in gps_data:
            rr.set_time_seconds("sensor_time", g['timestamp'] / (1000.0 * 1000.0))
            rr.set_time_seconds("z_gps_time", g['gps_timestamp_us'] / (1000.0 * 1000.0))
            rr.log("gps/points",
                   rr.GeoPoints(lat_lon=np.array([g['lat_rad'], g['lon_rad']])/ np.pi * 180.0,
                                radii=rr.Radius.ui_points(4.0), colors=[255,0,0]))
        # reset time so that we do not use gps_time for other stuff
        rr.reset_time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a V&R dataset with rerun.io")
    parser.add_argument("dir", type=str, help="Path to the dataset")
    parser.add_argument("--subsample_cam", type=int, default=10, help="Visualize every n-th image only")
    parser.add_argument("--send_laser_intensities", action=argparse.BooleanOptionalAction, help="Send laser intensities")
    args = parser.parse_args()
    if args.send_laser_intensities==None:
        args.send_laser_intensities = False

    rr.init("V&R Data Vis", spawn=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world/rig",
                name="3D Laser View",
                contents=["+ /world/rig/**", "+ /world/trajectory/**", "+ /world/**"],
                overrides={
                    "world/point_cloud": [
                        # rr.components.Color([0, 255, 0]),
                        # rr.datatypes.VisibleTimeRange("sensor_time",None,start=-5,end=5),
                    ]
                },
                # time_ranges=rrb.VisibleTimeRange("sensor_time", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-3), end=rrb.TimeRangeBoundary.cursor_relative(seconds=-3))
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="cam_1",
                        origin="/world/rig/cam_1/undistorted_image",
                        contents=[
                            "+ /world/rig/cam_1/**",
                        ],
                    ),
                    rrb.Spatial2DView(
                        name="cam_0",
                        origin="/world/rig/cam_0/undistorted_image",
                        contents=[
                            "+ /world/rig/cam_0/**",
                        ],
                    ),
                ),
                rrb.Tabs(
                    rrb.MapView(
                        origin="gps",
                        name="GPS Map View",
                        zoom=16.0,
                        background=rrb.MapProvider.OpenStreetMap,
                    ),
                    rrb.TimeSeriesView(
                        origin="/imu/accel",
                        name="Accel",
                        overrides={
                            "/imu/accel/x": [rr.components.Color([255, 0, 0])],
                            "/imu/accel/y": [rr.components.Color([0, 255, 0])],
                            "/imu/accel/z": [rr.components.Color([0, 0, 255])]
                        }
                    )
                    ,
                    rrb.TimeSeriesView(
                        origin="/imu/gyro",
                        name="Gyro",
                        overrides={
                            "/imu/gyro/x": [rr.components.Color([255, 0, 0])],
                            "/imu/gyro/y": [rr.components.Color([0, 255, 0])],
                            "/imu/gyro/z": [rr.components.Color([0, 0, 255])]
                        }
                    )
                )
            ),
            column_shares=[60, 40]
        )
    )

    rr.connect_tcp(default_blueprint=blueprint)
    rr.send_blueprint(blueprint, make_active=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
    rr.log("world", rr.Transform3D(translation=[0, 0, 0], quaternion=[0, 0, 0, 1]), static=True)

    # Read JSON file
    with open(args.dir + "/export/output.laz.sensor_export.json", "r") as f:
        sensor_export = json.load(f)
    rr_send_gps_data(sensor_export);
    rr_send_imu_data(sensor_export);

    with open(args.dir + "/export/output.laz.trajectory.json", 'r') as f:
        traj_data = json.load(f)

        rr_send_sensor_extrinsics(traj_data);
        rr_send_rig_trajectory(traj_data);

        dir = args.dir + "/raw/";
        image_buffer = get_dataset_image_buffer(traj_data, dir, subsample=args.subsample_cam);

        visualize_laz(args.dir + "/export/output.laz", traj_data, image_buffer, args.send_laser_intensities)

# stereo rectification

# ci0 = data['camera_infos']["cam_0"]
# ci1 = data['camera_infos']["cam_1"]

# # Camera intrinsic parameters
# fx_l, fy_l, cx_l, cy_l = ci0['fx'], ci0['fy'], ci0['cx'], ci0['cy']
# fx_r, fy_r, cx_r, cy_r = ci1['fx'], ci1['fy'], ci1['cx'], ci1['cy']

# camera_matrix_left = np.array([[fx_l, 0, cx_l],
#                             [0, fy_l, cy_l],
#                             [0, 0, 1]])

# camera_matrix_right = np.array([[fx_r, 0, cx_r],
#                                 [0, fy_r, cy_r],
#                                 [0, 0, 1]])

# dist_coeffs_left = np.array(ci0['dist_coeffs'])
# dist_coeffs_right = np.array(ci1['dist_coeffs'])

# R, t = relative_transform(data["sensor_to_trajectory_poses"]["cam_0"],data["sensor_to_trajectory_poses"]["cam_1"])

# # Image size
# image_size = (ci1['image_height'], ci1['image_width']) # swapped?

# # Stereo rectification
# R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix_left, dist_coeffs_left,
#                                             camera_matrix_right, dist_coeffs_right,
#                                             image_size, R, t)

# rr.log(f"world/rig/cam_0/rect", rr.Transform3D(translation=[pose[0],pose[1],pose[2]], mat3x3=rr.Quaternion(xyzw=[pose[4],pose[5],pose[6],pose[3]])), static=True)

# # Compute rectification maps
# map1_left, map2_left = cv2.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
# map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

# Apply rectification
# rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
# rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

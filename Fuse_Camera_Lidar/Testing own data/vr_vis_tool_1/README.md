# V&R Dataset Visualization Tool

This script allows you to visualize a V&R dataset using [rerun.io](https://rerun.io/). It loads images from a specified dataset folder and optionally subsamples them for efficient viewing.

## Installation

### 1. Set up a virtual environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
(Use .\.venv\Scripts\activate on Windows.)
```

### 2. Install dependencies

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### (3. Update dependencies when they change)

This is only required when new packages are installed. In this case, update requirements.txt:

```bash
pip freeze > requirements.txt
```

## Visualizing a dataset

Run the following command:

```bash
python vis.py <dataset_folder>
```

Replace <dataset_folder> with the path to your dataset.

Optional Parameters
The script supports an optional argument for subsampling images:

| Argument         | Type  | Default | Description |
|-----------------|------|---------|-------------|
| `dataset_folder`           | str  | Required | Path to the dataset folder. |
| `--subsample_cam` | int  | `10`     | Only visualize every n-th image (e.g., `--subsample_cam 5` shows every 5th image). |

## Example Usage

To visualize a dataset and only display every 5th image:

```bash
python vis.py /path/to/dataset --subsample_cam 5
```

## Notes

Ensure that the dataset folder contains images in a supported format. The folder layout should look like this:

```
2025_??_??_name_of_dataset/...
.../metadata.json
.../raw/cam_0/<cam_images>.jpg
.../raw/cam_1/<cam_images>.jpg
.../export/output.laz
.../export/output.laz.index.json
.../export/output.laz.sensor_export.json
.../export/output.laz.trajectory.json
```

## Data Format

The data format consists of four files that store various sensor outputs.

### 1. `export/output.laz`

This file is a `.laz` point cloud containing all laser points from both lidars, all referenced in a globally consistent 'world' coordinate system.

- The attribute `intensity` represents laser intensity and must be normalized by dividing by `65535`.

### 2. `export/output.laz.index.json`

This file provides an index to retrieve individual scans from the `.laz` file. It contains metadata about each scan, including its location within the `.laz` file.

#### Format

```json
{
    "__meta_data__": {
        "type": "LaserVOSlamResultExportIndex",
        "version": 1
    },
    "entries": [
        {
            "end_ts": 4099753,
            "laser_index": 0,
            "laser_name": "laser_0",
            "num_pts": 44708,
            "part_file_path": "output.laz",
            "part_index": 0,
            "point_offset_in_file": 0,
            "scan_index": 0,
            "start_ts": 3999741
        },
        ...
    ]
}
```

#### Fields

- **`end_ts`**: Last point timestamp of the scan (microseconds).
- **`laser_index`**: Index of the laser (0 for `laser_0`, 1 for `laser_1`).
- **`laser_name`**: Name of the laser (`laser_0` or `laser_1`).
- **`num_pts`**: Number of points in the scan.
- **`part_file_path`**: Always `output.laz` (reserved for future use).
- **`part_index`**: Always `0` (reserved for future use).
- **`point_offset_in_file`**: Point offset of the first point in the `.laz` file.
- **`scan_index`**: Index of the scan, incrementing per lidar revolution.
- **`start_ts`**: First point timestamp of the scan (microseconds).

### 3. `export/output.laz.trajectory.json`

This file contains trajectory information of the sensor rig.

#### Format

```json
{
    "__meta_data__": {
        "type": "TrajectoryExport",
        "version": 1
    },
    "camera_infos": {
        "cam_0": {
            "cx" : 1110.1763032586127,
            "cy" : 1128.0238482968696,
            "dist_coeffs" : [
                -0.07614366103756844, 0.06377132697082506, 2.147543913517001e-05, 0.00021198647620133856, 0.0
            ],
            "fx" : 1538.4885620293865,
            "fy" : 1538.1382479549222,
            "image_height" : 2252,
            "image_width" : 2248,
            "is_fisheye" : false,
            "timestamp_offset" : 100000,
            "timestamp_scale" : 1.0
        },
        "cam_1": { ... }
    },
    "geographic_coordinate_system": null,
    "sensor_to_trajectory_poses": {
        "laser_0": [ -0.02167, 0.01919, 0.08114, 0.99992, 0.01129, -0.00207, 0.00460 ],
        "laser_1": [ -0.03094, 0.94003, 0.26985, 0.00036, -0.00698, 0.27890, 0.96029 ]
    },
    "timestamp_system": "UC",
    "trajectory_poses": [
        [
            3999949,
            [0.0061, 0.0166, 0.0061, 0.9997, -0.0198, 0.0033, -0.0005]
        ],
        ...
    ]
}
```

#### Fields

- **`camera_infos`**: Contains camera calibration data in OpenCV format. Important: `timestamp_offset` needs to be added on all camera timestamps to obtain the correct timestamp of a frame.
- **`sensor_to_trajectory_poses`**: Defines sensor positions relative to the rig as a 7D pose `[x, y, z, qw, qx, qy, qz]` (unit quaternion for orientation).
- **`trajectory_poses`**: Contains timestamped 7D poses transforming rig coordinates into world coordinates.

### 4. `export/output.laz.sensor_export.json`

This file contains additional sensor data, including GPS and IMU measurements.

#### Format

```json
{
    "gps": [
        {
            "ecef_x": 2520545.37,
            "ecef_y": 811816.43,
            "ecef_z": 5783466.06,
            "fix_type": 2,
            "gps_timestamp_us": 1425818388000249,
            "height_ellipsoidal": 427.519,
            "lat_rad": 1.14395,
            "lon_rad": 0.31158,
            "timestamp": 3692200
        },
        ...
    ],
    "imu": [
        {
            "accel_x": -0.0710,
            "accel_y": -0.2376,
            "accel_z": 9.8154,
            "gyro_x": 0.00087,
            "gyro_y": 0.00218,
            "gyro_z": -0.00043,
            "timestamp": 1099206
        },
        ...
    ]
}
```

#### Fields

##### **GPS Data (`gps`)**

- **`ecef_x, ecef_y, ecef_z`**: ECEF (Earth-Centered, Earth-Fixed) coordinates.
- **`fix_type`**: GPS fix type (`0=None`, `1=GPS`, `2=DGPS`).
- **`gps_timestamp_us`**: GPS timestamp (microseconds).
- **`height_ellipsoidal`**: Ellipsoidal height (meters).
- **`lat_rad, lon_rad`**: Latitude and longitude (radians).
- **`timestamp`**: System timestamp (microseconds).

##### **IMU Data (`imu`)**

- **`accel_x, accel_y, accel_z`**: Acceleration (m/s²).
- **`gyro_x, gyro_y, gyro_z`**: Gyroscope readings (rad/s).
- **`timestamp`**: System timestamp (microseconds).

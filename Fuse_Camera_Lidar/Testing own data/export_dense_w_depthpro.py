#!/usr/bin/env python3

import os
import json
import re
import random
import cv2
import laspy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################

def pose7d_to_4x4(x, y, z, qw, qx, qy, qz):
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = rot
    T[:3, 3] = [x, y, z]
    return T

def invert_transform(T):
    R = T[:3,:3]
    t = T[:3,3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3,:3] = R.T
    T_inv[:3,3] = -R.T @ t
    return T_inv

def combine_transforms(T_child_parent, T_parent_grandparent):
    return T_parent_grandparent @ T_child_parent

def last_or_next_pose_up_to(timestamp, trajectory_poses):
    chosen = None
    for [ts, pose7d] in trajectory_poses:
        if ts <= timestamp:
            chosen = pose7d
        else:
            break
    if chosen is None:
        return trajectory_poses[0][1]
    return chosen

def undistort_image(image_bgr, fx, fy, cx, cy, dist_coeffs):
    h, w = image_bgr.shape[:2]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    und_img = cv2.undistort(image_bgr, K, dist)
    return und_img

def write_ply(filepath, points_xyz, colors_bgr=None):
    N = points_xyz.shape[0]
    has_color = (colors_bgr is not None) and (colors_bgr.shape[0] == N)
    with open(filepath,"w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x,y,z= points_xyz[i]
            if has_color:
                b,g,r= colors_bgr[i]
                r = int(max(0, min(255, r)))
                g = int(max(0, min(255, g)))
                b = int(max(0, min(255, b)))
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                f.write(f"{x} {y} {z}\n")

###############################################################################
# 2) RANSAC GROUND SEG
###############################################################################

def ransac_plane(points_xyz, sample_size=3, inlier_thresh=0.2, max_iter=1000):
    best_inliers_count= 0
    best_inliers_mask= None
    best_plane= None
    N= points_xyz.shape[0]
    if N<3:
        return None, np.zeros(N,dtype=bool)

    import random
    rng= np.random.default_rng(123)
    idx_arr= np.arange(N)

    def plane_from_3(p0,p1,p2):
        v1= p1-p0
        v2= p2-p0
        n= np.cross(v1,v2)
        norm= np.linalg.norm(n)
        if norm<1e-9:
            return None
        n= n/norm
        d= -np.dot(n, p0)
        return (n,d)

    for _ in range(max_iter):
        choice= rng.choice(idx_arr, size=3, replace=False)
        p0,p1,p2= points_xyz[choice]
        plane= plane_from_3(p0,p1,p2)
        if plane is None:
            continue
        n,d= plane
        dist= np.abs(points_xyz.dot(n)+d)
        inliers= (dist<=inlier_thresh)
        c_= np.count_nonzero(inliers)
        if c_> best_inliers_count:
            best_inliers_count= c_
            best_inliers_mask= inliers
            best_plane= (n,d)
    if best_plane is None:
        return None, np.zeros(N,dtype=bool)
    return best_plane, best_inliers_mask

###############################################################################
# 3) PARSE PER-LIDAR TIME WINDOWS
###############################################################################

def parse_lidar_time_windows(arg_list):
    """
    e.g. "laser_0=2,5" => { "laser_0": (2.0,5.0) }
    """
    twindows= {}
    for item in arg_list:
        # e.g. item="laser_0=2,5"
        if "=" not in item:
            continue
        name, range_str= item.split("=")
        if "," not in range_str:
            continue
        parts= range_str.split(",")
        if len(parts)!=2:
            continue
        try:
            tmin= float(parts[0])
            tmax= float(parts[1])
            twindows[name]= (tmin,tmax)
        except:
            print(f"WARNING: parse time window => {item}")
    return twindows

###############################################################################
# 4) DEPTHPRO HELPER
###############################################################################

def load_depthpro_if_needed(use_depthpro):
    """
    If user wants --compare_depthpro, load or create DepthPro model once here.
    """
    if not use_depthpro:
        return None
    import depth_pro  # You must ensure 'depth_pro' is installable
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    return (model, transform)

def run_depthpro_on_image(model_tuple, image_path, K_fx):
    """
    model_tuple = (model, transform) from 'depth_pro'.
    K_fx: focal length (approx) if needed by DepthPro
    returns a 2D depth array, shape (H,W)
    """
    import depth_pro
    import torch
    from PIL import Image

    model, transform = model_tuple
    # load
    image_rgb, intrinsics, f_px = depth_pro.load_rgb(image_path)
    input_tensor = transform(image_rgb)
    with torch.no_grad():
        prediction = model.infer(input_tensor, f_px=f_px)
    depth = prediction["depth"].cpu().numpy()
    return depth

def scale_depth_with_median_lidar(aggregated_world_camZ, aggregated_px_py,
                                  depthpro_raw, valid_mask, threshold=10.0):
    """
    - aggregated_world_camZ: array of shape (N,) camera-space Z for aggregator points
    - aggregated_px_py: array of shape (N,2) => (px_i, py_i)
    - depthpro_raw: 2D array => DepthPro raw output
    - valid_mask: boolean array (N,) => aggregator points that pass near-lidar, dynamic, etc.
    - threshold= some max difference to skip outliers for ratio computing

    Return a single scale_factor. If no inliers => scale=1.0
    """
    inliers_ratio = []
    H, W = depthpro_raw.shape
    for i in range(len(valid_mask)):
        if not valid_mask[i]:
            continue
        px, py = aggregated_px_py[i]
        if px<0 or px>=W or py<0 or py>=H:
            continue
        z_lidar = aggregated_world_camZ[i]
        z_pred  = depthpro_raw[py, px]
        if z_lidar>0.1 and z_pred>0.1:
            ratio = z_lidar / z_pred
            # skip if difference is too large
            if abs(z_lidar - z_pred) < threshold:
                inliers_ratio.append(ratio)
    if len(inliers_ratio)<5:
        return 1.0
    median_scale = np.median(inliers_ratio)
    return median_scale

###############################################################################
# 5) MAIN
###############################################################################

def main():
    import argparse
    parser= argparse.ArgumentParser(description="""
    Combined script with:
      - per-lidar time windows
      - near-lidar dist skip
      - aggregator-based dynamic optional
      - ground seg optional
      - DepthPro pseudo-depth check => skip aggregator points >1.0m mismatch
    """)
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--camera_name", default="cam_0")
    parser.add_argument("--image_index", type=int, default=50)
    parser.add_argument("--lidar_names", nargs="+", default=["laser_0","laser_1"])
    parser.add_argument("--lidar_time_windows", nargs="+", default=[])
    parser.add_argument("--lidar_filter_distance", nargs="+", default=[])
    parser.add_argument("--filter_dynamic", action="store_true")
    parser.add_argument("--dynamic_dist_thresh", type=float, default=0.3)
    parser.add_argument("--segment_ground", action="store_true")
    parser.add_argument("--ground_inlier_thresh", type=float, default=0.2)
    parser.add_argument("--compare_depthpro", action="store_true",
                        help="If set, run DepthPro on the image. Scale it, skip aggregator points > 1m mismatch.")
    parser.add_argument("--depthpro_diff_threshold", type=float, default=1.0,
                        help="Points with |z_lidar - z_depthPro|> this are removed.")
    parser.add_argument("--out_ply", type=str, default="")

    args= parser.parse_args()

    ds= args.dataset_dir
    cam_name= args.camera_name
    idx_img= args.image_index

    # parse time windows
    twindows= parse_lidar_time_windows(args.lidar_time_windows)
    print("Time windows =>", twindows)

    # parse filter dist
    filter_dict={}
    for p in args.lidar_filter_distance:
        if "=" not in p:
            continue
        ln, valstr= p.split("=")
        try:
            valf= float(valstr)
            filter_dict[ln]= valf
        except:
            pass
    print("Lidar dist =>", filter_dict)

    # load trajectory
    traj_path= os.path.join(ds,"export","output.laz.trajectory.json")
    with open(traj_path,"r") as f:
        traj_data= json.load(f)
    if cam_name not in traj_data["camera_infos"]:
        print("camera not found => abort.")
        return
    cam_info= traj_data["camera_infos"][cam_name]
    fx, fy= cam_info["fx"], cam_info["fy"]
    cx, cy= cam_info["cx"], cam_info["cy"]
    dist_coeffs= cam_info["dist_coeffs"]
    offset= cam_info.get("timestamp_offset",0)
    scale= cam_info.get("timestamp_scale",1.0)
    s2t_poses= traj_data["sensor_to_trajectory_poses"]
    rig_trajectory= traj_data["trajectory_poses"]

    # pick image
    raw_cam_dir= os.path.join(ds,"raw",cam_name)
    pat= re.compile(rf"{cam_name}_(\d+)\.jpg")
    images=[]
    for fn in os.listdir(raw_cam_dir):
        m= pat.match(fn)
        if m:
            itime= int(m.group(1))
            images.append((itime, os.path.join(raw_cam_dir,fn)))
    images.sort(key=lambda x:x[0])

    if idx_img<0 or idx_img>= len(images):
        print(f"index out of range => {idx_img}/{len(images)}")
        return

    img_ts_raw, img_path= images[idx_img]
    cam_ts= int(img_ts_raw* scale + offset)
    print(f"Chosen image => idx={idx_img}, {img_path}, cam_ts={cam_ts}")

    # load & undistort
    img_bgr= cv2.imread(img_path)
    if img_bgr is None:
        print("fail load =>", img_path)
        return
    und_img= undistort_image(img_bgr, fx, fy, cx, cy, dist_coeffs)
    imH, imW= und_img.shape[:2]

    # camera->world
    rig_pose7d= last_or_next_pose_up_to(cam_ts, rig_trajectory)
    T_rig_world= pose7d_to_4x4(*rig_pose7d)
    T_cam_rig_7d= s2t_poses[cam_name]
    T_cam_rig= pose7d_to_4x4(*T_cam_rig_7d)
    T_cam_world= combine_transforms(T_cam_rig, T_rig_world)
    T_world_cam= invert_transform(T_cam_world)

    # if user wants DepthPro => load model once
    depthpro_model_tuple= None
    if args.compare_depthpro:
        try:
            import depth_pro
        except ImportError:
            print("ERROR: depth_pro not installed => cannot compare DepthPro. Aborting the feature.")
            args.compare_depthpro= False
        else:
            from PIL import Image
            depthpro_model_tuple= load_depthpro_if_needed(True)

    # read .laz index
    laz_path= os.path.join(ds,"export","output.laz")
    with open(laz_path+".index.json","r") as f:
        idxd= json.load(f)
    entries= idxd["entries"]

    def overlap_range(s_ts,e_ts, amin,amax):
        return (e_ts>=amin) and (s_ts<=amax)

    aggregator_points= np.zeros((0,3), dtype=np.float32)
    all_scans_world= []

    with laspy.open(laz_path) as lasf:
        for e in entries:
            ln= e["laser_name"]
            if ln not in args.lidar_names:
                continue
            if ln not in twindows:
                # skip if no time window for this sensor
                continue
            (tw_min, tw_max)= twindows[ln]
            abs_min= cam_ts + int(tw_min*1e6)
            abs_max= cam_ts + int(tw_max*1e6)
            s_ts= e["start_ts"]
            e_ts= e["end_ts"]
            if not overlap_range(s_ts,e_ts, abs_min,abs_max):
                continue

            # build LiDAR->world for chunk
            chunk_ts= e_ts
            rig_pose7d2= last_or_next_pose_up_to(chunk_ts, rig_trajectory)
            T_rig_world2= pose7d_to_4x4(*rig_pose7d2)
            if ln not in s2t_poses:
                continue
            T_lidar_rig_7d= s2t_poses[ln]
            T_lidar_rig= pose7d_to_4x4(*T_lidar_rig_7d)
            T_lidar_world= combine_transforms(T_lidar_rig, T_rig_world2)
            T_world_lidar= invert_transform(T_lidar_world)

            minDist= filter_dict.get(ln, 0.0)
            lasf.seek(e["point_offset_in_file"])
            pts= lasf.read_points(e["num_pts"])
            coords_world= np.vstack((pts.x, pts.y, pts.z)).T

            # skip near-lidar
            if minDist>0.0:
                N0= coords_world.shape[0]
                hom_= np.hstack([coords_world, np.ones((N0,1), coords_world.dtype)])
                coords_lidar= (T_world_lidar@ hom_.T).T
                dist_l= np.sqrt( np.sum(coords_lidar[:,:3]**2, axis=1) )
                keep_mask= (dist_l>= minDist)
                coords_world= coords_world[keep_mask]

            # aggregator-based dynamic
            if args.filter_dynamic and aggregator_points.shape[0]>0 and coords_world.shape[0]>0:
                kd_agg= cKDTree(aggregator_points)
                dists, _= kd_agg.query(coords_world, k=1, workers=-1)
                keep_mask= (dists<= args.dynamic_dist_thresh)
                coords_world= coords_world[keep_mask]

            if coords_world.shape[0]>0:
                aggregator_points= np.vstack([aggregator_points, coords_world])
                all_scans_world.append(coords_world)

    if len(all_scans_world)==0:
        print("No points => done.")
        return

    aggregated_world= np.concatenate(all_scans_world, axis=0)
    print(f"Aggregated => {aggregated_world.shape[0]} total points")

    # 5) If user wants DepthPro => run it => do median scale => discard big mismatch
    if args.compare_depthpro:
        if depthpro_model_tuple is not None:
            # run depthpro
            from PIL import Image
            import torch
            # 5a) get pseudo-depth raw
            import depth_pro
            K_fx = fx  # approximate focal length to pass to DepthPro
            depthpro_raw= run_depthpro_on_image(depthpro_model_tuple, img_path, K_fx=K_fx)

            # 5b) we need aggregator in camera coords => to compute scale
            Nf= aggregated_world.shape[0]
            if Nf>0:
                homf= np.hstack([aggregated_world, np.ones((Nf,1), aggregated_world.dtype)])
                T_world_cam= invert_transform(T_cam_world)
                coords_cam= (T_world_cam @ homf.T).T
                Xc= coords_cam[:,0]
                Yc= coords_cam[:,1]
                Zc= coords_cam[:,2]

                # front mask => Z>0
                front_mask= (Zc>0.1)
                px_f= fx*(Xc/np.maximum(Zc,1e-6)) + cx
                py_f= fy*(Yc/np.maximum(Zc,1e-6)) + cy
                px_i= px_f.astype(int)
                py_i= py_f.astype(int)

                # 5c) find a median scale factor
                # We'll do a small function inline:
                inliers_ratio= []
                Hdp, Wdp= depthpro_raw.shape
                for i in range(Nf):
                    if not front_mask[i]:
                        continue
                    ix, iy= px_i[i], py_i[i]
                    if ix<0 or ix>=Wdp or iy<0 or iy>=Hdp:
                        continue
                    zL= Zc[i]
                    zD= depthpro_raw[iy, ix]
                    # skip if zL or zD < 0.1 => near camera or invalid
                    if zL>0.1 and zD>0.1:
                        ratio= zL / zD
                        # skip if huge difference => 10m
                        if abs(zL - zD)<10.0:
                            inliers_ratio.append(ratio)
                if len(inliers_ratio)<5:
                    scale_factor= 1.0
                else:
                    scale_factor= np.median(inliers_ratio)
                print(f"DepthPro scale_factor => {scale_factor:.3f}")

                # 5d) apply scale => scaled_depth= scale_factor* depthpro_raw
                depthpro_scaled= depthpro_raw* scale_factor

                # 5e) discard aggregator points if |Zc - depthpro_scaled[py,px]|> threshold
                mismatch_mask= np.ones((Nf,), dtype=bool)  # True => keep
                for i in range(Nf):
                    if not front_mask[i]:
                        mismatch_mask[i]= False
                        continue
                    ix, iy= px_i[i], py_i[i]
                    if ix<0 or ix>=Wdp or iy<0 or iy>=Hdp:
                        mismatch_mask[i]= False
                        continue
                    zL= Zc[i]
                    zD= depthpro_scaled[iy, ix]
                    # if difference>1 => skip
                    if abs(zL - zD)> args.depthpro_diff_threshold:
                        mismatch_mask[i]= False
                # rebuild aggregator
                aggregated_world= aggregated_world[mismatch_mask]
                Zc= Zc[mismatch_mask]  # might be used in next steps
                print(f"DepthPro => removed {np.count_nonzero(~mismatch_mask)} points => remain {aggregated_world.shape[0]}")

    if aggregated_world.shape[0]==0:
        print("No points remain => after DepthPro => done.")
        return

    # 6) ground seg => if needed
    if args.segment_ground and aggregated_world.shape[0]>3:
        plane, inliers= ransac_plane(aggregated_world,
                                     sample_size=3,
                                     inlier_thresh=args.ground_inlier_thresh,
                                     max_iter=1000)
        if plane is not None:
            keep_mask= ~inliers
            removed= np.count_nonzero(inliers)
            aggregated_world= aggregated_world[keep_mask]
            print(f"Removed ground => {removed}, remain={aggregated_world.shape[0]}")

    if aggregated_world.shape[0]==0:
        print("No points remain => done.")
        return

    # 7) final camera overlay => z-buffer
    Nf= aggregated_world.shape[0]
    hom_= np.hstack([aggregated_world, np.ones((Nf,1), aggregated_world.dtype)])
    T_world_cam= invert_transform(T_cam_world)
    coords_cam= (T_world_cam@ hom_.T).T
    Xc= coords_cam[:,0]
    Yc= coords_cam[:,1]
    Zc= coords_cam[:,2]
    front_mask= (Zc>1e-6)

    px_f= fx*(Xc/ np.maximum(Zc,1e-6)) + cx
    py_f= fy*(Yc/ np.maximum(Zc,1e-6)) + cy
    px_i= px_f.astype(int)
    py_i= py_f.astype(int)

    z_buffer= {}
    imH, imW= und_img.shape[:2]
    for i in range(Nf):
        if not front_mask[i]:
            continue
        ix, iy= px_i[i], py_i[i]
        if ix<0 or ix>= imW or iy<0 or iy>= imH:
            continue
        depth= Zc[i]
        if (ix, iy) not in z_buffer:
            z_buffer[(ix, iy)] = (i, depth)
        else:
            (pi,pd)= z_buffer[(ix, iy)]
            if depth< pd:
                z_buffer[(ix, iy)] = (i, depth)
    chosen_idx= np.array([v[0] for v in z_buffer.values()], dtype=int)
    print(f"Z-buffer => kept {len(chosen_idx)}/{Nf}")

    # color from camera
    import copy
    final_xyz= aggregated_world[chosen_idx]
    colors_bgr= np.zeros((len(chosen_idx),3), dtype=np.uint8)
    for j,i in enumerate(chosen_idx):
        ix, iy= px_i[i], py_i[i]
        colors_bgr[j]= und_img[iy, ix]

    debug_img= copy.deepcopy(und_img)
    for j,i in enumerate(chosen_idx):
        ix, iy= px_i[i], py_i[i]
        cv2.circle(debug_img,(ix,iy),2,(0,255,0),-1)

    h2,w2= debug_img.shape[:2]
    sc= min(1280.0/w2,720.0/h2) if (w2>0 and h2>0) else 1.0
    if sc<1.0:
        debug_img= cv2.resize(debug_img,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
    cv2.imshow("Final overlay",debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 8) out ply
    if args.out_ply:
        print(f"Writing {len(final_xyz)} => {args.out_ply}")
        write_ply(args.out_ply, final_xyz, colors_bgr)

if __name__=="__main__":
    main()

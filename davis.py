import argparse
import subprocess
import shutil

from concurrent import futures

import open3d as o3d
import imageio

import json
import numpy as np
import imageio
import glob
import moviepy as mvp
from moviepy.editor import *

import os
import os.path as osp
from utils.image_io import load_raw_float32_image
from utils.visualization import visualize_depth_dir


# Change based on your output path.
DAVIS_DIR = "/home/vickie/data/DAVIS/JPEGImages/480p"
DAVIS_DIR = "/home/vickie/data/posetrack/images"
OUTPUT_ROOT = "outputs"


def copy_frames(seq):
    output_dir = os.path.join(OUTPUT_ROOT, seq)
    os.makedirs(output_dir, exist_ok=True)

    src_dir = os.path.join(DAVIS_DIR, seq)
    ext = os.path.splitext(os.listdir(src_dir)[0])[-1].lower()
    src_paths = sorted(glob.glob(f"{src_dir}/*{ext}"))

    tgt_dir = os.path.join(output_dir, "color_full")
    tgt_paths = sorted(glob.glob(f"{tgt_dir}/*.png"))

    if len(tgt_paths) != len(src_paths):
        cmd = f"cp -r {src_dir} {tgt_dir}"
        print(cmd)
        subprocess.call(cmd, shell=True)

        cwd = os.getcwd()
        os.chdir(tgt_dir)
        cmd = f"rename 's/^/frame_/' *{ext}"
        print(cmd)
        subprocess.call(cmd, shell=True)
        os.chdir(cwd)

        tgt_paths = sorted(glob.glob(f"{tgt_dir}/*{ext}"))
        assert len(src_paths) == len(tgt_paths)

        if ext != ".png":
            cmd = f"mogrify -format png {tgt_dir}/*{ext}"
            print(cmd)
            subprocess.call(cmd, shell=True)
            tgt_paths = sorted(glob.glob(f"{tgt_dir}/*.png"))
            assert len(src_paths) == len(tgt_paths)

            cmd = f"rm {tgt_dir}/*{ext}"
            print(cmd)
            subprocess.call(cmd, shell=True)
            print(f"converted {ext} from {src_dir} to png in {tgt_dir}")

    pts_file = os.path.join(output_dir, "frames.txt")
    im = imageio.imread(tgt_paths[0])
    H, W = im.shape[:2]
    N = len(tgt_paths)
    lines = [N, H, W] + list(range(N))
    with open(pts_file, "w") as f:
        f.write("\n".join(map(str, lines)))
    print("pts info written to {}".format(pts_file))


RES_ROOT = "R_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0"
DEPTH_OUT = "depth_e0000/e0000_filtered/depth"


def run_opt(seq, gpu, overwrite=False):
    output_dir = os.path.join(OUTPUT_ROOT, seq)
    input_dir = os.path.join(output_dir, "color_full")
    input_paths = glob.glob(f"{input_dir}/*.png")
    depth_dir = os.path.join(output_dir, RES_ROOT, DEPTH_OUT)
    depth_paths = glob.glob(f"{depth_dir}/*.raw")

    if not overwrite and len(depth_paths) == len(input_paths):
        print(f"already {len(depth_paths)}/{len(input_paths)} written")
        return

    cmd = "python main.py --path {}".format(output_dir)
    args = "--save_intermediate_depth_streams_freq 1 \
            --num_epochs 0 \
            --post_filter \
            --opt.adaptive_deformation_cost 10 \
            --save_depth_visualization \
            --save_tensorboard"
    cmd = "CUDA_VISIBLE_DEVICES={} {} {}".format(gpu, cmd, args)
    print(cmd)
    subprocess.call(cmd, shell=True)


def visualize_results(seq, fps=10):
    output_dir = os.path.join(OUTPUT_ROOT, seq)

    depth_midas_dir = osp.join(output_dir, "depth_midas2/depth")
    depth_vis_midas_dir = osp.join(output_dir, "depth_vis_midas2")
    os.makedirs(depth_vis_midas_dir, exist_ok=True)
    visualize_depth_dir(depth_midas_dir, depth_vis_midas_dir)

    depth_result_dir = osp.join(output_dir, RES_ROOT, DEPTH_OUT)
    depth_vis_result_dir = osp.join(output_dir, "depth_vis_result")
    os.makedirs(depth_vis_result_dir, exist_ok=True)
    visualize_depth_dir(depth_result_dir, depth_vis_result_dir)

    color_dir = osp.join(output_dir, "color_down_png")
    clip_color = ImageSequenceClip(color_dir, fps=fps)
    clip_midas = ImageSequenceClip(depth_vis_midas_dir, fps=fps)
    clip_result = ImageSequenceClip(depth_vis_result_dir, fps=fps)

    clip_color = clip_color.set_duration(clip_result.duration)
    clip_midas = clip_midas.set_duration(clip_result.duration)

    clip_color.write_videofile(osp.join(output_dir, "clip_color.mp4"), fps=fps)
    clip_midas.write_videofile(osp.join(output_dir, "clip_midas.mp4"), fps=fps)
    clip_result.write_videofile(osp.join(output_dir, "clip_result.mp4"), fps=fps)

    video_color = VideoFileClip(osp.join(output_dir, "clip_color.mp4"))
    video_midas = VideoFileClip(osp.join(output_dir, "clip_midas.mp4"))
    video_result = VideoFileClip(osp.join(output_dir, "clip_result.mp4"))

    video = clips_array([[video_color, video_midas, video_result]])
    video.write_videofile(
        osp.join(output_dir, "video_comparison.mp4"), fps=fps, codec="mpeg4"
    )


def get_xy_grid(W, H):
    return np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)


def vis_3d(seq):
    out_dir = os.path.join(OUTPUT_ROOT, seq)
    res_dir = os.path.join(out_dir, RES_ROOT)

    inp_dir = os.path.join(out_dir, "color_down_png")
    inp_paths = glob.glob(f"{inp_dir}/*.png")
    N = len(inp_paths)

    # load depth images
    depth_dir = os.path.join(res_dir, DEPTH_OUT)
    depth_paths = sorted(glob.glob(f"{depth_dir}/*.raw"))
    assert len(depth_paths) == N

    # load camera info
    intrinsics = np.loadtxt(os.path.join(res_dir, "intrinsics.txt"))
    fx, fy, cx, cy = np.split(intrinsics, 4, axis=-1)
    poses = np.loadtxt(os.path.join(res_dir, "extrinsics.txt"))
    assert len(poses) == N
    poses = poses.reshape((N, 3, 4))

    # (N, 3, 3), (N, 3, 1)
    R_c2w, t_c2w = poses[:, :3, :3], poses[:, :3, 3:]
    R_w2c = np.transpose(R_c2w, [0, 2, 1])  # (N, 3, 3)
    t_w2c = -np.matmul(R_w2c, t_c2w)  # (N, 3, 1)

    test = imageio.imread(inp_paths[0])
    H, W = test.shape[:2]
    xy = get_xy_grid(W, H)
    print(xy.shape, H, W)

    pcls = []
    for i in range(N):
        rgb = imageio.imread(inp_paths[i]) / 255
        depth = 1.0 / load_raw_float32_image(depth_paths[i])
        valid = np.isfinite(depth)  # (H, W, 1)
        depth = depth.reshape((H, W, 1))
        xyz_c = np.concatenate([xy, depth], axis=-1)  # (H, W, 3)
        R = R_w2c[i].reshape(1, 1, 3, 3)
        t = t_w2c[i].reshape(1, 1, 3, 1)
        xyz_w = np.matmul(R, xyz_c[..., None]) + t
        pcl = np.concatenate([xyz_w[valid, :, 0], rgb[valid]], axis=-1)
        pcls.append(pcl.astype(np.float32))

    pcls = np.concatenate(pcls, axis=0)

    fused = np2o3d(pcls)
    pcl_path = os.path.join(out_dir, "res_fused.ply")
    o3d.io.write_point_cloud(pcl_path, fused)
    print("wrote point cloud to", pcl_path)

    # write cameras in json
    cam_path = os.path.join(out_dir, "res_cameras.json")
    cam_dict = {
        "rotation": R_c2w.reshape((N, 9)).tolist(),
        "translation": t_c2w.reshape((N, 3)).tolist(),
        "intrinsics": intrinsics.tolist(),
    }
    with open(cam_path, "w") as f:
        json.dump(cam_dict, f)
    print("wrote cameras to", cam_path)


def np2o3d(arr):
    assert arr.ndim > 1
    d = arr.shape[-1]
    assert d == 3 or d == 6
    arr = arr.reshape((-1, d))
    points = o3d.utility.Vector3dVector(arr[:, :3])
    pcl = o3d.geometry.PointCloud(points)
    if d == 6:
        pcl.colors = o3d.utility.Vector3dVector(arr[:, 3:])
    return pcl


def process_sequence(seq, gpu):
    print("processing", seq)
    #     copy_frames(seq)
    #     run_opt(seq, gpu)
    vis_3d(seq)


#     visualize_results(seq)


if __name__ == "__main__":
    n_gpus = 1
    seqs = sorted(os.listdir(DAVIS_DIR))
    seqs = ["024159_mpii_test"]

    for i, seq in enumerate(seqs):
        gpu = i % n_gpus
        process_sequence(seq, gpu)

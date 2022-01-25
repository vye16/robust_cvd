import argparse
import subprocess
import shutil

from concurrent import futures

import imageio
import glob
import moviepy as mvp
from moviepy.editor import *

import os
import os.path as osp
from utils.visualization import visualize_depth_dir


# Change based on your output path.
DAVIS_DIR = "/home/vickie/data/DAVIS/JPEGImages/480p"
OUTPUT_ROOT = "outputs"


def copy_frames(seq):
    output_dir = os.path.join(OUTPUT_ROOT, seq)
    os.makedirs(output_dir, exist_ok=True)

    src_dir = os.path.join(DAVIS_DIR, seq)
    tgt_dir = os.path.join(output_dir, "color_full")
    shutil.copy(src_dir, tgt_dir)
    src_paths = sorted(glob.glob("{}/*.jpg".format(src_dir)))
    jpg_paths = sorted(glob.glob("{}/*.jpg".format(tgt_dir)))
    assert len(src_paths) == len(jpg_paths)

    cmd = "mogrify -format png {}/*.jpg".format(tgt_dir)
    subprocess.call(cmd, shell=True)
    png_paths = sorted(glob.glob("{}/*.png".format(tgt_dir)))
    assert len(png_paths) == len(jpg_paths)

    subprocess.call("rm {}/*.jpg".format(tgt_dir))
    print("converted jpg from {} to png in {}".format(src_dir, tgt_dir))

    pts_file = os.path.join(output_dir, "frames.txt")
    im = imageio.imread(jpg_paths[0])
    H, W = im.shape[:2]
    N = len(jpg_paths)
    lines = [N, H, W] + list(range(N))
    with open(pts_file, "w") as f:
        f.write("\n".join(map(str, lines)))
    print("pts info written to {}".format(pts_file))


def run_opt(seq, gpu):
    output_dir = os.path.join(OUTPUT_ROOT, seq)
    cmd = "python main.py --path {}".format(output_dir)
    args = "--save_intermediate_depth_streams_freq 1 \
            --num_epochs 0 \
            --post_filter \
            --opt.adaptive_deformation_cost 10 \
            --save_depth_visualization"
    cmd = "CUDA_VISIBLE_DEVICES={} {} {}".format(gpu, cmd, args)
    print(cmd)
    subprocess.call(cmd, shell=True)


def visualize_results(seq, fps=10):
    output_dir = os.path.join(OUTPUT_ROOT, seq)

    depth_midas_dir = osp.join(output_dir, "depth_midas2/depth")
    depth_vis_midas_dir = osp.join(output_dir, "depth_vis_midas2")
    os.makedirs(depth_vis_midas_dir, exist_ok=True)
    visualize_depth_dir(depth_midas_dir, depth_vis_midas_dir)

    depth_result_dir = osp.join(
        output_dir,
        "R_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/e0000_filtered/depth/",
    )
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


def process_sequence(seq, gpu):
    print("processing", seq)
    copy_frames(seq)
    run_opt(seq, gpu)
    visualize_results(seq)


if __name__ == "__main__":
    n_gpus = 2
    seqs = sorted(os.listdir(DAVIS_DIR))

    for i, seq in enumerate(seqs):
        gpu = i % n_gpus
        process_sequence(seq, gpu)

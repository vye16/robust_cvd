import os
import sys

from lib_python import DepthVideo, DepthVideoImporter


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", "-r", type=str, required=True)
    args = parser.parse_args()

    # Load depth video.
    video = DepthVideo()
    discoverStreams = True
    DepthVideoImporter.importVideo(video, args.results_dir, discoverStreams)
    video.printInfo()
    import ipdb; ipdb.set_trace()

#!/bin/bash
python2 dataloaders/associate.py $1/depth.txt $1/rgb.txt >> $1/associations.txt
mkdir $1/pose_per_frame
python3 pre_process/store_matched_pose_tensors.py $1/associations.txt $1/groundtruth.txt $1/pose_per_frame
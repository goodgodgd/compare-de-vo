#! /bin/bash

source ../settings.sh

python ./kitti/generate_pose_snippets.py \
	--dataset_dir="$KITTI_ODOM_RAW" \
	--output_dir="$KITTI_ODOM_STACKED" \
	--seq_id=09 \
	--seq_length=5


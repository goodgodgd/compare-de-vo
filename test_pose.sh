#! /bin/bash

source ./settings.sh

python devo_bench_main.py \
	--mode=test_pose \
	--dataset_dir="$KITTI_ODOM_RAW" \
	--init_ckpt_file="$KITTI_ODOM_CKPT" \
	--batch_size=1 \
	--seq_length=5 \
	--pose_test_seq=9 \
	--output_dir="$KITTI_ODOM_PREDICTION"


#! /bin/bash

source ./settings.sh

python devo_bench_main.py \
	--mode="test_pose" \
	--dataset_dir="$KITTI_ODOM_STACKED" \
	--tfrecords_dir="$KITTI_ODOM_TFRECORD" \
	--init_ckpt_file="$POSE_NET_MODEL/model" \
	--checkpoint_dir="$POSE_NET_MODEL" \
	--batch_size=32 \
	--seq_length=5 \
	--output_dir="$KITTI_ODOM_PREDICT"


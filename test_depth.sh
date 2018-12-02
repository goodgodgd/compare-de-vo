#! /bin/bash

source ./settings.sh

python devo_bench_main.py \
	--mode="test_depth" \
	--dataset_dir="$KITTI_EIGEN_STACKED" \
	--tfrecords_dir="$KITTI_EIGEN_TFRECORD" \
	--init_ckpt_file="$DEPTH_NET_MODEL/model" \
	--checkpoint_dir="$DEPTH_NET_MODEL" \
	--batch_size=1 \
	--output_dir="$KITTI_EIGEN_PREDICT"


#! /bin/bash

source ./settings.sh

python devo_bench_main.py \
	--mode=test_pose \
	--dataset_dir="$KITTI_POSE_RAW" \
	--init_ckpt_file="/home/ian/workspace/CompareDeVo/ckpts/geonet_posenet" \
	--batch_size=1 \
	--seq_length=5 \
	--pose_test_seq=9 \
	--output_dir="$KITTI_POSE_PREDICTION"


#!/bin/bash

source ../settings.sh

python3 tfrecord_main.py \
	--dataset_dir="$KITTI_EIGEN_STACKED" \
	--dataset_name=kitti_raw_eigen \
	--dump_root="$KITTI_EIGEN_TFRECORD" \
	--seq_length=3


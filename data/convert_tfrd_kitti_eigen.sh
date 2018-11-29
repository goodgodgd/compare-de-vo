#!/bin/bash

source ../settings.sh

python3 tfrecord_main.py \
	--dataset_dir="$KITTI_DEPTH_STACKED" \
	--dataset_name=kitti_raw_eigen \
	--dump_root="$KITTI_TEST_TFRECORD"


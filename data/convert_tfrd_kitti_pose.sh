#!/bin/bash

source ../settings.sh

python3 tfrecord_example.py \
	--dataset_dir="$KITTI_ODOM_STACKED" \
	--dataset_name=kitti_odom \
	--dump_root="$KITTI_ODOM_TRRECORD"


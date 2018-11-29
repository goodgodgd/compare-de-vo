#! /bin/bash

source ../settings.sh

python prepare_train_data.py \
	--dataset_dir="$KITTI_ODOM_RAW" \
	--dataset_name=kitti_odom \
	--dump_root="$KITTI_ODOM_STACKED" \
	--seq_length=5 \
	--img_height=128 \
	--img_width=416 \
	--num_threads=8 \
	--remove_static 

#! /bin/bash

source ../settings.sh

python prepare_stacked_data.py \
	--dataset_dir="$KITTI_EIGEN_RAW" \
	--dataset_name=kitti_raw_eigen \
	--dump_root="$KITTI_EIGEN_STACKED" \
	--seq_length=3 \
	--img_height=128 \
	--img_width=416 \
	--num_threads=16 \
	--remove_static 


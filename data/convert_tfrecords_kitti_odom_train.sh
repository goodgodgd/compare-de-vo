#!/bin/bash

python3 tfrecord_example.py \
	--dataset_dir=/media/ian/iandata/geonet_data/kitti_odom \
	--dataset_name=kitti_odom \
	--dump_root=/media/ian/iandata/geonet_data/tfrecords/kitti_odom \
	--img_height=128 \
	--img_width=2080


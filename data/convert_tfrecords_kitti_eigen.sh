#!/bin/bash

python3 tfrecord_example.py \
	--dataset_dir=/media/ian/iandata/geonet_data/kitti_raw_eigen \
	--dataset_name=kitti_raw_eigen \
	--dump_root=/media/ian/iandata/geonet_data/tfrecords/kitti_raw_eigen

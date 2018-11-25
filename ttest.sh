#! /bin/bash

source ./settings.sh

python ttest_geonet_feeder.py \
	--mode=train_rigid \
	--tfrecords_dir="$KITTI_DEPTH_TFRECORD" \
	--checkpoint_dir="../ckpt" \
	--learning_rate=0.0002 \
	--seq_length=3 \
	--batch_size=4 \
	--train_epochs=50 


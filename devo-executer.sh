#! /bin/bash

# where "kitti_raw_data" and "kitti_odom" exist
RAW_DATA_ROOT="/media/ian/My Passport"
# where all the results saved
OUTPUT_PATH="/home/ian/workplace/CompareDevo/devo_bench_data"

KITTI_ODOM_RAW="$RAW_DATA_ROOT/data_odometry"
KITTI_ODOM_STACKED="$OUTPUT_PATH/kitti_odom"
KITTI_ODOM_TFRECORD="$OUTPUT_PATH/tfrecords/kitti_odom"
POSE_NET_MODEL="$OUTPUT_PATH/ckpts/geonet/geonet_posenet"

KITTI_EIGEN_RAW="$RAW_DATA_ROOT/kitti_raw_data"
KITTI_EIGEN_STACKED="$OUTPUT_PATH/kitti_raw_eigen"
KITTI_EIGEN_TFRECORD="$OUTPUT_PATH/tfrecords/kitti_raw_eigen"
DEPTH_NET_MODEL="$OUTPUT_PATH/ckpts/geonet/geonet_depthnet"

PREDICT_OUTPUT="$OUTPUT_PATH/predicts"

if [ "$1" == "--help" ]
then
	echo "devo-executer [option]"
	echo "Options:"
	echo "prepare_paths : make required dirs under OUTPUT_PATH"
	echo "prepare_kitti_eigen :	prepare stacked images and gt depth data from kitti raw dataset"
	echo "prepare_kitti_odom : prepare stacked images and gt pose data from kitti odom dataset"
	echo "make_tfrecord_eigen : create tfrecord files from the results of prepare_kitti_eigen"
	echo "make_tfrecord_odom : create tfrecord files from the results of prepare_kitti_odom"
	echo "train_rigid : train pose and depth prediction model"
	echo "pred_depth : predict depths from test data and save them"
	echo "pred_pose : predict poses from test data and save them"

elif [ "$1" == "prepare_paths" ]
then
	if [ ! -d "$RAW_DATA_ROOT" ]
	then
		echo [Error] "$RAW_DATA_ROOT" does not exits!
		exit 0
	elif [ ! -d "$OUTPUT_PATH" ]
	then
		echo [Error] "$OUTPUT_PATH" does not exits!
		exit 0
	fi
	
	mkdir -p "$KITTI_ODOM_RAW"
	mkdir -p "$KITTI_ODOM_STACKED"
	mkdir -p "$KITTI_ODOM_TFRECORD"
	mkdir -p "$POSE_NET_MODEL"

	mkdir -p "$KITTI_EIGEN_RAW"
	mkdir -p "$KITTI_EIGEN_STACKED"
	mkdir -p "$KITTI_EIGEN_TFRECORD"
	mkdir -p "$DEPTH_NET_MODEL"

	mkdir -p "$PREDICT_OUTPUT"

elif [ "$1" == "prepare_kitti_eigen" ]
then
	python data/prepare_stacked_data.py \
		--dataset_dir="$KITTI_EIGEN_RAW" \
		--dataset_name=kitti_raw_eigen \
		--dump_root="$KITTI_EIGEN_STACKED" \
		--seq_length=3 \
		--img_height=128 \
		--img_width=416 \
		--num_threads=8 \
		--remove_static 

elif [ "$1" == "prepare_kitti_odom" ]
then
	python data/prepare_stacked_data.py \
		--dataset_dir="$KITTI_ODOM_RAW" \
		--dataset_name=kitti_odom \
		--dump_root="$KITTI_ODOM_STACKED" \
		--seq_length=5 \
		--img_height=128 \
		--img_width=416 \
		--num_threads=8 \
		--remove_static 

elif [ "$1" == "make_tfrecord_eigen" ]
then
	python3 data/tfrecord_main.py \
		--dataset_dir="$KITTI_EIGEN_STACKED" \
		--dataset_name="kitti_raw_eigen" \
		--dump_root="$KITTI_EIGEN_TFRECORD" \
		--seq_length=3

elif [ "$1" == "make_tfrecord_odom" ]
then
	python3 data/tfrecord_main.py \
		--dataset_dir="$KITTI_ODOM_STACKED" \
		--dataset_name=kitti_odom \
		--dump_root="$KITTI_ODOM_TFRECORD" \
		--seq_length=5

elif [ "$1" == "train_rigid" ]
then
	python devo_bench_main.py \
		--mode=train_rigid \
		--tfrecords_dir="$KITTI_ODOM_TFRECORD" \
		--checkpoint_dir="$NEW_TRAIN_MODEL" \
		--learning_rate=0.0002 \
		--seq_length=3 \
		--batch_size=4 \
		--train_epochs=50 

elif [ "$1" == "pred_depth" ]
then
	python devo_bench_main.py \
		--mode="pred_depth" \
		--dataset_dir="$KITTI_EIGEN_STACKED" \
		--tfrecords_dir="$KITTI_EIGEN_TFRECORD" \
		--init_ckpt_file="$DEPTH_NET_MODEL/model" \
		--checkpoint_dir="$DEPTH_NET_MODEL" \
		--batch_size=1 \
		--output_dir="$PREDICT_OUTPUT"

elif [ "$1" == "pred_pose" ]
then
	python devo_bench_main.py \
		--mode="pred_pose" \
		--dataset_dir="$KITTI_ODOM_STACKED" \
		--tfrecords_dir="$KITTI_ODOM_TFRECORD" \
		--init_ckpt_file="$POSE_NET_MODEL/model" \
		--checkpoint_dir="$POSE_NET_MODEL" \
		--batch_size=32 \
		--output_dir="$PREDICT_OUTPUT"

elif [ "$1" == "eval_depth" ]
then
	python devo_bench_main.py \
		--mode="eval_depth" \
		--output_dir="$PREDICT_OUTPUT"

elif [ "$1" == "eval_pose" ]
then
	python devo_bench_main.py \
		--mode="eval_pose" \
		--output_dir="$PREDICT_OUTPUT"

else
	echo "invalid option"

fi


#! /bin/bash

### MANUAL SET: where "kitti_raw_data" and "kitti_odom" exist
RAW_DATA_ROOT="/media/ian/iandata"
### MANUAL SET: where all the results saved
OUTPUT_PATH="/media/ian/iandata/devo_bench_data"
### MANUAL SET: model name
MODEL_NAME="geonet_inct4"
### MANUAL SET: encoder name ["resnet50, "inceptionv4", "vgg16"]
ENCODER="inceptionv4"

KITTI_ODOM_RAW="$RAW_DATA_ROOT/kitti_odometry"
KITTI_ODOM_STACKED="$OUTPUT_PATH/kitti_odom"
KITTI_ODOM_TFRECORD="$OUTPUT_PATH/tfrecords/kitti_odom"

KITTI_EIGEN_RAW="$RAW_DATA_ROOT/kitti_raw_data"
KITTI_EIGEN_STACKED="$OUTPUT_PATH/kitti_raw_eigen"
KITTI_EIGEN_TFRECORD="$OUTPUT_PATH/tfrecords/kitti_raw_eigen"

MODEL_CKPT_DIR="$OUTPUT_PATH/ckpts/$MODEL_NAME"
NEW_TRAIN_MODEL="$MODEL_CKPT_DIR/train"
PREDICT_OUTPUT="$OUTPUT_PATH/predicts"
EVALUATION_OUTPUT="$OUTPUT_PATH/evaluation"

### MANUAL SET: checkpoints for evaluation
POSE_EVAL_CKPT="$MODEL_CKPT_DIR/posenet/model"
DEPTH_EVAL_CKPT="$MODEL_CKPT_DIR/depthnet/model"


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
	
	mkdir -p "$KITTI_ODOM_STACKED"
	mkdir -p "$KITTI_ODOM_TFRECORD"
	mkdir -p "$KITTI_EIGEN_STACKED"
	mkdir -p "$KITTI_EIGEN_TFRECORD"

	mkdir -p "$MODEL_CKPT_DIR"
	mkdir -p "$PREDICT_OUTPUT"
	mkdir -p "$EVALUATION_OUTPUT"

elif [ "$1" == "prepare_kitti_eigen" ]
then
	python data/prepare_stacked_data.py \
		--raw_dataset_dir="$KITTI_EIGEN_RAW" \
		--dataset_name=kitti_raw_eigen \
		--dump_root="$KITTI_EIGEN_STACKED" \
		--seq_length=3 \
		--split="all" \
		--img_height=128 \
		--img_width=416 \
		--num_threads=8 \
		--remove_static 

elif [ "$1" == "prepare_kitti_odom" ]
then
	python data/prepare_stacked_data.py \
		--raw_dataset_dir="$KITTI_ODOM_RAW" \
		--dataset_name=kitti_odom \
		--dump_root="$KITTI_ODOM_STACKED" \
		--seq_length=5 \
		--split="all" \
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
		--mode="train_rigid" \
		--model_name="$MODEL_NAME" \
		--net_encoder="$ENCODER" \
		--tfrecords_dir="$KITTI_ODOM_TFRECORD" \
		--checkpoint_dir="$NEW_TRAIN_MODEL" \
		--learning_rate=0.0002 \
		--seq_length=5 \
		--batch_size=8 \
		--train_epochs=50 

elif [ "$1" == "pred_depth" ]
then
	python devo_bench_main.py \
		--mode="pred_depth" \
		--model_name="$MODEL_NAME" \
		--tfrecords_dir="$KITTI_EIGEN_TFRECORD" \
		--init_ckpt_file="$DEPTH_EVAL_CKPT" \
		--seq_length=3 \
		--batch_size=1 \
		--pred_out_dir="$PREDICT_OUTPUT"

elif [ "$1" == "pred_pose" ]
then
	python devo_bench_main.py \
		--mode="pred_pose" \
		--model_name="$MODEL_NAME" \
		--tfrecords_dir="$KITTI_ODOM_TFRECORD" \
		--init_ckpt_file="$POSE_EVAL_CKPT" \
		--seq_length=5 \
		--batch_size=4 \
		--pred_out_dir="$PREDICT_OUTPUT"

elif [ "$1" == "eval_depth" ]
then
	python devo_bench_main.py \
		--mode="eval_depth" \
		--pred_out_dir="$PREDICT_OUTPUT" \
		--eval_out_dir="$EVALUATION_OUTPUT"

elif [ "$1" == "eval_pose" ]
then
	python devo_bench_main.py \
		--mode="eval_pose" \
		--pred_out_dir="$PREDICT_OUTPUT" \
		--eval_out_dir="$EVALUATION_OUTPUT" \
		--seq_length=5

elif [ "$1" == "eval_traj" ]
then
	python devo_bench_main.py \
		--mode="eval_traj" \
		--pred_out_dir="$PREDICT_OUTPUT" \
		--eval_out_dir="$EVALUATION_OUTPUT" \
		--seq_length=5

else
	echo "invalid option, please type ./devo-executer.sh --help"

fi


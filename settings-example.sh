#! /bin/bash

PRJ_PATH="/home/ian/workplace/CompareDevo"
DATA_DRIVE="/media/ian/My Passport"
REOR_DATA="$PRJ_PATH/geonet_data"

KITTI_ODOM_RAW="$DATA_DRIVE/data_odometry"
KITTI_ODOM_STACKED="$REOR_DATA/kitti_odom"
KITTI_ODOM_TFRECORD="$REOR_DATA/tfrecords/kitti_odom"
KITTI_ODOM_PREDICT="$REOR_DATA/predicts/kitti_odom"
POSE_NET_MODEL="$PRJ_PATH/ckpts/geonet_posenet"

KITTI_EIGEN_RAW="$DATA_DRIVE/kitti_raw_data"
KITTI_EIGEN_STACKED="$REOR_DATA/kitti_raw_eigen"
KITTI_EIGEN_TFRECORD="$REOR_DATA/tfrecords/kitti_raw_eigen"
KITTI_EIGEN_PREDICT="$REOR_DATA/predicts/kitti_raw_eigen"
DEPTH_NET_MODEL="$PRJ_PATH/ckpts/geonet_depthnet"


#! /bin/bash

DATA_DRIVE="/media/ian/My Passport"

KITTI_ODOM_RAW="$DATA_DRIVE/data_odometry_color"
KITTI_ODOM_STACKED="$DATA_DRIVE/geonet_data/kitti_odom"
KITTI_ODOM_TFRECORD="$DATA_DRIVE/geonet_data/tfrecords/kitti_odom"
KITTI_ODOM_PREDICT="$DATA_DRIVE/geonet_data/predicts/kitti_odom"

KITTI_DEPTH_RAW="$DATA_DRIVE/??"
KITTI_DEPTH_STACKED="$DATA_DRIVE/geonet_data/kitti_raw_eigen"
KITTI_DEPTH_TFRECORD="$DATA_DRIVE/geonet_data/tfrecords/kitti_raw_eigen"
KITTI_DEPTH_PREDICT="$DATA_DRIVE/geonet_data/predicts/kitti_raw_eigen"


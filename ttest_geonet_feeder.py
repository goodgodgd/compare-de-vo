# TODO: 코드 출처 표기
import os
import random
import pprint
import numpy as np
import tensorflow as tf

from models.geonet.geonet_feeder import dataset_feeder


flags = tf.app.flags
flags.DEFINE_string("model",                        "",    "geonet or sfmlearner")
flags.DEFINE_string("mode",              "train_rigid",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("tfrecords_dir",  "/media/ian/My Passport/geonet_data/tfrecords/kitti_raw_eigen",    "tfrecords directory")
flags.DEFINE_string("init_ckpt_file",             None,    "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4,    "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32,    "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_integer("seq_length",                   3,    "Sequence length for each example")

# #### Training Configurations #####
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002,    "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",                 20,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("train_epochs",                50,    "number of epochs for training")

flags.DEFINE_integer("num_source",                   0,    "number of sources")
flags.DEFINE_integer("num_scales",                   4,    "number of scales")
flags.DEFINE_integer("add_flownet",                  0,    "whether flownet is included in model")
flags.DEFINE_integer("add_dispnet",                  0,    "whether dispnet is included in model")
flags.DEFINE_integer("add_posenet",                  0,    "whether posenet is included in model")

opt = flags.FLAGS


def main(_):
    tf.enable_eager_execution()
    print("eager execution")
    dataset = dataset_feeder(opt, "train")
    print("dataset", dataset)
    for features in dataset:
        src_image_stack = features["sources"]
        tgt_image = features["target"]
        intrinsics_ms = features["intrinsics_ms"]
        print("shapes", src_image_stack.shape, tgt_image.shape, intrinsics_ms.shape)


if __name__ == '__main__':
    tf.app.run()

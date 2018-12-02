import os
import sys
import tensorflow as tf

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path: sys.path.append(module_path)
from models.geonet.geonet_feeder import dataset_feeder
from models.geonet.geonet_model import GeoNetModel
from model_operator import GeoNetOperator


flags = tf.app.flags
flags.DEFINE_string("mode",              "train_rigid",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("tfrecords_dir",                "",    "tfrecords directory")

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
flags.DEFINE_float("alpha_recon_image",           0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")

# #### Additional Configurations #####
flags.DEFINE_integer("num_source",                   2,    "number of sources")
flags.DEFINE_integer("num_scales",                   4,    "number of scales")
flags.DEFINE_integer("add_flownet",                  0,    "whether flownet is included in model")
flags.DEFINE_integer("add_dispnet",                  1,    "whether dispnet is included in model")
flags.DEFINE_integer("add_posenet",                  1,    "whether posenet is included in model")

# #### Configurations about DepthNet & PoseNet of GeoNet #####
flags.DEFINE_string("dispnet_encoder",      "resnet50",    "Type of encoder for dispnet, vgg or resnet50")
flags.DEFINE_boolean("scale_normalize",          False,    "Spatially normalize depth prediction")
flags.DEFINE_float("rigid_warp_weight",            1.0,    "Weight for warping by rigid flow")
flags.DEFINE_float("disp_smooth_weight",           0.5,    "Weight for disp smoothness")

# #### Configurations about ResFlowNet of GeoNet (or DirFlowNetS) #####
flags.DEFINE_string("flownet_type",         "residual",    "type of flownet, residual or direct")
flags.DEFINE_float("flow_warp_weight",             1.0,    "Weight for warping by full flow")
flags.DEFINE_float("flow_smooth_weight",           0.2,    "Weight for flow smoothness")
flags.DEFINE_float("flow_consistency_weight",      0.2,    "Weight for bidirectional flow consistency")
flags.DEFINE_float("flow_consistency_alpha",       3.0,    "Alpha for flow consistency check")
flags.DEFINE_float("flow_consistency_beta",       0.05,    "Beta for flow consistency check")

opt = flags.FLAGS


def main(_):
    tf.enable_eager_execution()
    print("enable eager execution")

    opt.mode = "test_depth"

    if opt.mode == "test_pose":
        opt.dataset_dir = "/home/ian/workplace/CompareDevo/geonet_data/kitti_odom"
        opt.tfrecords_dir = "/home/ian/workplace/CompareDevo/geonet_data/tfrecords/kitti_odom"
        opt.checkpoint_dir = "/home/ian/workplace/CompareDevo/ckpts/geonet_posenet"
        opt.batch_size = 4
        opt.seq_length = 5
        opt.num_source = 4

    if opt.mode == "test_depth":
        opt.dataset_dir = "/home/ian/workplace/CompareDevo/geonet_data/kitti_raw_eigen"
        opt.tfrecords_dir = "/home/ian/workplace/CompareDevo/geonet_data/tfrecords/kitti_raw_eigen"
        opt.checkpoint_dir = "/home/ian/workplace/CompareDevo/ckpts/geonet_depthnet"
        opt.batch_size = 4
        opt.seq_length = 1
        opt.num_source = 2

    dataset = dataset_feeder(opt, "test", opt.seq_length)
    # geonet = GeoNetModel(opt)
    # model_op = GeoNetOperator(opt, geonet)

    if tf.executing_eagerly():
        for i, features in enumerate(dataset):
            src_image_stack = features["sources"]
            tgt_image = features["target"]
            gtruth = features["gt"]
            intrinsics_ms = features["intrinsics_ms"]

            print("=========== features")
            print("srcimg:", src_image_stack.shape)
            print("tgtimg:", tgt_image.shape)
            print("intrin:", intrinsics_ms.shape)
            print("gtruth:", gtruth.shape)

            if i >= 10:
                break


if __name__ == '__main__':
    tf.app.run()

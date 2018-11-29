import sys
import tensorflow as tf
sys.path.append('..')
from models.geonet.geonet_feeder import dataset_feeder
from models.geonet.geonet_model import GeoNetModel
from model_operator import GeoNetOperator


flags = tf.app.flags
flags.DEFINE_string("mode",              "train_rigid",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("tfrecords_dir",                "",    "tfrecords directory")
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")

flags.DEFINE_integer("batch_size",                   4,    "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32,    "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_integer("seq_length",                   3,    "Sequence length for each example")

flags.DEFINE_integer("train_epochs",                50,    "number of epochs for training")

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
    print("eager execution")
    opt.dataset_dir = "/media/ian/iandata/geonet_data/kitti_raw_eigen"
    opt.tfrecords_dir = "/media/ian/iandata/geonet_data/tfrecords/kitti_raw_eigen"
    opt.checkpoint_dir = "../"
    opt.mode = "test_depth"
    dataset = dataset_feeder(opt, "train")
    print("dataset", dataset)

    geonet = GeoNetModel(opt)
    model_op = GeoNetOperator(opt, geonet)

    for features in dataset:
        src_image_stack = features["sources"]
        tgt_image = features["target"]
        intrinsics_ms = features["intrinsics_ms"]
        print("shapes", src_image_stack.shape, tgt_image.shape, intrinsics_ms.shape)
        model_op._cnn_model_fn(features, tf.estimator.ModeKeys.PREDICT)
        break


if __name__ == '__main__':
    tf.app.run()

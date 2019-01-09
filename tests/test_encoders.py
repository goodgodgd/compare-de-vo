import os
import sys
import tensorflow as tf

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path: sys.path.append(module_path)
from data.tfrecord_feeder import dataset_feeder


flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (pred_depth, pred_pose, test_flow)")
flags.DEFINE_string("model_name",                   "",    "geonet or sfmlearner")
flags.DEFINE_string("dataset_name",             "KITTI",    "KITTI")
flags.DEFINE_string("tfrecords_dir",                "",    "tfrecords directory")
flags.DEFINE_string("eval_out_dir",                 "",    "evaluation result directory")
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
flags.DEFINE_float("alpha_recon_image",           0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")
flags.DEFINE_integer("save_ckpt_freq",            5000,    "Save the checkpoint model every save_ckpt_freq iterations")

# #### Configurations about DepthNet & PoseNet of GeoNet #####
flags.DEFINE_string("feat_net",      "resnet50",    "Type of encoder for dispnet, vgg or resnet50")
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

# #### Testing Configurations #####
flags.DEFINE_string("pred_out_dir",                 None,    "Test result output directory")
flags.DEFINE_string("depth_test_split",        "eigen",    "KITTI depth split, eigen or stereo")

# #### Evaluation Configurations #####
flags.DEFINE_float("min_depth",                   1e-3,    "Threshold for minimum depth")
flags.DEFINE_float("max_depth",                     80,    "Threshold for maximum depth")

# #### Additional Configurations #####
flags.DEFINE_integer("num_source",                   0,    "number of sources")
flags.DEFINE_integer("num_scales",                   0,    "number of scales")
flags.DEFINE_integer("add_flownet",                  0,    "whether flownet is included in model")
flags.DEFINE_integer("add_dispnet",                  0,    "whether dispnet is included in model")
flags.DEFINE_integer("add_posenet",                  0,    "whether posenet is included in model")

opt = flags.FLAGS


def set_dependent_opts():
    # set subordinative variables
    if opt.mode == "train_rigid":
        opt.seq_length = 3
    elif opt.mode in ["pred_pose", "eval_pose", "eval_traj"]:
        opt.seq_length = 5
    elif opt.mode == "pred_depth":
        opt.seq_length = 1
        opt.batch_size = 1
    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'pred_depth']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'pred_pose']

    if "KITTI" in opt.dataset_name:
        opt.img_height = 128
        opt.img_width = 416


def create_model():
    net_model = None
    if opt.mode in ['train_rigid', 'pred_depth', 'pred_pose']:
        print("modelname", opt.model_name)
        if opt.model_name == "geonet":
            print("geonet")
            from models.geonet.geonet_model import GeoNetModel
            net_model = GeoNetModel(opt)
        elif opt.model_name == "geonet_inct4":
            print("geonet_inception")
            from models.geonet_inct4.geonet_inct4_model import GeoNetInct4Model
            net_model = GeoNetInct4Model(opt)
    return net_model


def main(_):
    opt.model_name = "geonet_inct4"
    data_root = "/media/ian/iandata/vode_bench_data"
    opt.mode = "train_rigid"
    opt.tfrecords_dir = os.path.join(data_root, "tfrecords", "kitti_odom")
    opt.checkpoint_dir = os.path.join(data_root, "ckpts", opt.model_name, "train")
    opt.seq_length = 5
    opt.batch_size = 4
    opt.train_epochs = 50
    set_dependent_opts()
    print("important opt", "\ntfrecord", opt.tfrecords_dir, "\ncheckpoint", opt.checkpoint_dir,
          "\nbatch", opt.batch_size, ", img_height", opt.img_height, ", img_width", opt.img_width,
          "\ntrain_epoch", opt.train_epochs, ", learning reate", opt.learning_rate,
          "\nnum_scales", opt.num_scales, "\nseq_length", opt.seq_length)

    features = dataset_feeder(opt, "train")
    src_image_stack = features["sources"]
    tgt_image = features["target"]
    gtruth = features["gt"]
    intrinsics_ms = features["intrinsics_ms"]
    print("tgt_image.shape", tgt_image.get_shape())

    geonet = create_model()
    geonet.build_model(tgt_image, src_image_stack, intrinsics_ms)


if __name__ == '__main__':
    tf.app.run()

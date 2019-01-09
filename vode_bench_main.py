# TODO: code reference
import tensorflow as tf
import traintesteval as tte
import importlib

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (pred_depth, pred_pose, test_flow)")
flags.DEFINE_string("model_name",              "geonet",    "geonet")
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
flags.DEFINE_string("feat_net",             "resnet50",    "Type of encoder for dispnet, vgg or resnet50")
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
    if opt.mode == "pred_depth":
        opt.seq_length = 1
        opt.batch_size = 1
    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'pred_depth']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'pred_pose']

    if "kitti" in opt.dataset_name.lower():
        opt.img_height = 128
        opt.img_width = 416

    print("========== important options \ntfrecords_dirs: {} \ncheckpoint_dir: {}\n".
            format(opt.tfrecords_dir, opt.checkpoint_dir),
          "batch_size={}, img_height={}, img_width={}, seq_length={}\n".
            format(opt.batch_size, opt.img_height, opt.img_width, opt.seq_length),
          "train_epoch={}, learning_rate={}".format(opt.train_epochs, opt.learning_rate))


def create_model():
    net_model = None
    if opt.mode in ['train_rigid', 'pred_depth', 'pred_pose']:
        print("modelname", opt.model_name)
        if "geonet" in opt.model_name:
            from models.geonet.geonet_model import GeoNetModel
            net_model = GeoNetModel(opt)
        elif opt.model_name == "sfmlearner":
            from models.sfmlearner.SfMLearner import SfMLearner
            net_model = SfMLearner(opt)
    return net_model


def main(_):
    set_dependent_opts()
    net_model = create_model()
    print("net model", net_model)

    if opt.mode == 'train_rigid':
        tte.train(opt, net_model)
    elif opt.mode == 'pred_depth':
        tte.pred_depth(opt, net_model)
    elif opt.mode == 'pred_pose':
        tte.pred_pose(opt, net_model)
    elif opt.mode == 'eval_depth':
        tte.eval_depth(opt)
    elif opt.mode == 'eval_pose':
        tte.eval_pose(opt)
    elif opt.mode == 'eval_traj':
        tte.eval_traj(opt)


if __name__ == '__main__':
    tf.app.run()

import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

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

flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002,    "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",                 20,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("train_epochs",                50,    "number of epochs for training")
flags.DEFINE_integer("num_source",                   0,    "number of sources")
flags.DEFINE_integer("num_scales",                   0,    "number of scales")

opt = flags.FLAGS


def build_model(input_layer):
    input_layer = tf.cast(input_layer, dtype=tf.float32)
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        activation_fn=tf.nn.relu):
        net = slim.conv2d(inputs=input_layer, kernel_size=[5, 5], num_outputs=32, scope='conv1',
                          padding='same')
        net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
        net = slim.conv2d(inputs=net, kernel_size=[5, 5], num_outputs=64, scope='conv2', padding='same')
        net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
        net = slim.flatten(inputs=net, scope='flatten')
        net = slim.fully_connected(inputs=net, num_outputs=1024, scope='fc3')
    print("`````reg losses", tf.losses.get_regularization_losses())
    return net


if __name__ == '__main__':
    # !! Note !!: in eager_execution mode, tf.losses.get_regularization_losses() returns empty list
    # but when using tf.Estimator with eager_execution, it somehow works

    # tf.enable_eager_execution()

    data_root = "/media/ian/iandata/devo_bench_data"
    opt.mode = "train_rigid"
    opt.model_name = "geonet"
    opt.tfrecords_dir = os.path.join(data_root, "tfrecords", "kitti_odom")
    opt.checkpoint_dir = os.path.join(data_root, "ckpts", opt.model_name, "train")
    opt.seq_length = 5
    opt.batch_size = 4
    opt.train_epochs = 50
    opt.img_height = 128
    opt.img_width = 416
    opt.num_scales = 4
    opt.num_source = opt.seq_length - 1

    if tf.executing_eagerly():
        dataset = dataset_feeder(opt, "train")
        for i, features in enumerate(dataset):
            src_image_stack = features["sources"]
            tgt_image = features["target"]
            gtruth = features["gt"]
            intrinsics_ms = features["intrinsics_ms"]

            print("========== feature shape ==========")
            print("srcimg:", src_image_stack.shape)
            print("tgtimg:", tgt_image.shape)
            print("intrin:", intrinsics_ms.shape)
            print("gtruth:", gtruth.shape)

            net = build_model(tgt_image)
            print(net.get_shape())
            break
    else:
        features = dataset_feeder(opt, "train")
        # input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 3],
        #                              name='raw_input')
        src_image_stack = features["sources"]
        tgt_image = features["target"]
        gtruth = features["gt"]
        intrinsics_ms = features["intrinsics_ms"]

        net = build_model(tgt_image)
        print(net.get_shape())


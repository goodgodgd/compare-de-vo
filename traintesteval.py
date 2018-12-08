import os
import random
import numpy as np
import tensorflow as tf
import glob
import cv2
import pandas as pd


from models.geonet.geonet_model import GeoNetModel
from data.tfrecord_feeder import dataset_feeder
from model_operator import GeoNetOperator
from constants import InputShape
from data.kitti.kitti_pose_utils import save_pose_result, format_pose_seq_TUM, compute_pose_error
from data.kitti.kitti_depth_utils import compute_depth_errors, save_gt_depths, save_pred_depths


# ========== TRAIN ==========
def train(opt):
    set_random_seed()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    def data_feeder():
        return dataset_feeder(opt, "train", opt.seq_length)
    geonet = GeoNetModel(opt)
    model_op = GeoNetOperator(opt, geonet)
    model_op.train(data_feeder)


def set_random_seed():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ========== TEST ==========
def pred_pose(opt):
    geonet = GeoNetModel(opt)
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, InputShape.HEIGHT, InputShape.WIDTH,
                                            opt.seq_length * 3], name='raw_input')
    tgt_image = input_uint8[:, :, :, :3]
    src_image_stack = input_uint8[:, :, :, 3:]
    geonet.build_model(tgt_image, src_image_stack, None)
    fetches = {"pose": geonet.get_pose_pred()}
    saver = tf.train.Saver([var for var in tf.model_variables()])
    dataset_iter = dataset_feeder(opt, "test", opt.seq_length)

    gt_poses = []
    pred_poses = []
    frames = []
    target_ind = (opt.seq_length - 1)//2

    with tf.Session() as sess:
        saver.restore(sess, opt.init_ckpt_file)
        for i in range(1000000):
            try:
                inputs = sess.run(dataset_iter)
                pred = sess.run(fetches, feed_dict={tgt_image: inputs["target"],
                                                    src_image_stack: inputs["sources"]})
                frame_batch = [bytes(frame).decode("utf-8") for frame in inputs["frame_int8"]]
                frames.extend(frame_batch)
                gt_pose_batch = inputs["gt"]
                pred_pose_batch = pred["pose"]
                pred_pose_batch = np.insert(pred_pose_batch, target_ind, np.zeros((1, 6)), axis=1)
                for b in range(opt.batch_size):
                    pred_pose_tum = format_pose_seq_TUM(pred_pose_batch[b, :, :], gt_pose_batch[b, :, 0])
                    pred_poses.append(pred_pose_tum)
                    gt_poses.append(gt_pose_batch[b, :, :])
            except tf.errors.OutOfRangeError:
                print("dataset finished at step", i*opt.batch_size)
                break

    print("output length (gt, pred)", len(gt_poses), len(pred_poses))
    # one can evaluate pose errors here but we save results and evaluate it in the evaluation step
    save_pose_result(pred_poses, opt.output_dir, "geonet", frames, opt.seq_length)
    # save_pose_result(gt_poses, opt.output_dir, "ground_truth", frames, opt.seq_length)


def pred_depth(opt):
    geonet = GeoNetModel(opt)
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, InputShape.HEIGHT, InputShape.WIDTH,
                                            opt.seq_length * 3], name='raw_input')
    tgt_image = input_uint8[:, :, :, :3]
    geonet.build_model(tgt_image, None, None)
    fetches = {"depth": geonet.get_depth_pred()}
    saver = tf.train.Saver([var for var in tf.model_variables()])
    dataset_iter = dataset_feeder(opt, "test", opt.seq_length)

    gt_depths = []
    pred_depths = []

    with tf.Session() as sess:
        saver.restore(sess, opt.init_ckpt_file)
        for i in range(1000000):
            try:
                inputs = sess.run(dataset_iter)
                pred = sess.run(fetches, feed_dict={tgt_image: inputs["target"]})
                gt_depths.append(np.squeeze(inputs["gt"], 0))
                pred_depths.append(np.squeeze(pred["depth"], 3))
            except tf.errors.OutOfRangeError:
                print("dataset finished at step", i)
                break

    print("depths shape (gt, pred)", gt_depths[0].shape, pred_depths[0].shape)
    save_gt_depths(gt_depths, opt.output_dir)
    save_pred_depths(pred_depths, opt.output_dir, "geonet")


# ========== EVALUATION ==========
def eval_depth(opt):
    gt_dir = "ground_truth"
    model_names = os.listdir(opt.output_dir)
    model_names.remove(gt_dir)
    gt_path = os.path.join(opt.output_dir, gt_dir, "depth")
    pred_paths = {model: os.path.join(opt.output_dir, model, "depth") for model in model_names}

    for model, pred_path in pred_paths.items():
        gt_files = glob.glob(os.path.join(gt_path, "*.npy"))
        gt_files.sort()
        pred_file = os.path.join(pred_path, "kitti_eigen_depth_predictions.npy")
        if not os.path.isfile(pred_file):
            print("No depth prediction file from model {}".format(model))
            continue

        predicted_depths = np.load(pred_file)
        print(model, "predicted shape", predicted_depths.shape)
        evaluate_depth_errors(predicted_depths, gt_files, opt.min_depth, opt.max_depth)


def evaluate_depth_errors(pred_depths, gt_files, min_depth, max_depth):
    num_test = len(gt_files)
    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)

    for i, gt_file in enumerate(gt_files):
        gt_depth = np.load(gt_file)
        gt_height, gt_width = gt_depth.shape
        predi_depth = np.copy(pred_depths[i, :, :])
        predi_depth = cv2.resize(predi_depth, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(predi_depth[mask])
        predi_depth[mask] *= scalor

        predi_depth[predi_depth < min_depth] = min_depth
        predi_depth[predi_depth > max_depth] = max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_depth_errors(gt_depth[mask], predi_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}"
          .format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}"
          .format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(),
                  a1.mean(), a2.mean(), a3.mean()))


def eval_pose(opt):
    eval_seq = ["09", "10"]
    gt_dir = "ground_truth"
    model_names = os.listdir(opt.output_dir)
    model_names.remove(gt_dir)
    gt_path = os.path.join(opt.output_dir, gt_dir, "pose")
    pred_paths = {model: os.path.join(opt.output_dir, model, "pose") for model in model_names}
    dfcols = ["model", "seq", "index", "trn_err"]
    pose_err = pd.DataFrame(columns=dfcols)

    for model, pred_path in pred_paths.items():
        if not os.path.isdir(os.path.join(gt_path, eval_seq[0])):
            print("No pose prediction sequences from model {}".format(model))
            continue

        for seq in eval_seq:
            gt_files = glob.glob(os.path.join(gt_path, seq, "*.txt"))
            gt_files.sort()

            for i, gtfile in enumerate(gt_files):
                predfile = gtfile.replace(gt_path, pred_path)
                assert os.path.isfile(gtfile), "{} not found!!".format(gtfile)
                if not os.path.isfile(predfile):
                    continue

                gt_short_seq = np.loadtxt(gtfile)
                pred_short_seq = np.loadtxt(predfile)
                if pred_short_seq.ndim < 2:
                    continue

                assert (gt_short_seq.shape == (5, 8) and pred_short_seq.shape[1] == 8)
                te = compute_pose_error(gt_short_seq, pred_short_seq)
                pose_err = pose_err.append(dict(zip(dfcols, [model, seq, i, te])), ignore_index=True)

    print("pose errors {}\n".format(len(pose_err)), pose_err.head())

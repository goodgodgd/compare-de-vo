import os
import random
import numpy as np
import tensorflow as tf
import glob
import cv2
import pandas as pd

from data.tfrecord_feeder import dataset_feeder
import data.kitti.kitti_pose_utils as pu
import data.kitti.kitti_depth_utils as du


# ========== TRAIN ==========
def train(opt, model_op):
    set_random_seed()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    def data_feeder():
        return dataset_feeder(opt, "train")
    model_op.train(data_feeder)


def set_random_seed():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ========== TEST ==========
def pred_pose(opt, net_model):
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width,
                                            opt.seq_length * 3], name='raw_input')
    tgt_image = input_uint8[:, :, :, :3]
    src_image_stack = input_uint8[:, :, :, 3:]
    net_model.build_model(tgt_image, src_image_stack, None)
    fetches = {"pose": net_model.get_pose_pred()}
    saver = tf.train.Saver([var for var in tf.model_variables()])
    dataset_iter = dataset_feeder(opt, "test")

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
                    pred_pose_tum = pu.format_pose_seq_TUM(pred_pose_batch[b, :, :], gt_pose_batch[b, :, 0])
                    pred_poses.append(pred_pose_tum)
                    gt_poses.append(gt_pose_batch[b, :, :])
            except tf.errors.OutOfRangeError:
                print("dataset finished at step", i*opt.batch_size)
                break

    print("output length (gt, pred)", len(gt_poses), len(pred_poses))
    # one can evaluate pose errors here but we save results and evaluate it in the evaluation step
    pu.save_pose_result(pred_poses, opt.pred_out_dir, opt.model_name, frames, opt.seq_length)
    # save_pose_result(gt_poses, opt.pred_out_dir, "ground_truth", frames, opt.seq_length)


def pred_depth(opt, net_model):
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width,
                                            opt.seq_length * 3], name='raw_input')
    tgt_image = input_uint8[:, :, :, :3]
    net_model.build_model(tgt_image, None, None)
    fetches = {"depth": net_model.get_depth_pred()}
    saver = tf.train.Saver([var for var in tf.model_variables()])
    dataset_iter = dataset_feeder(opt, "test")

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
    du.save_gt_depths(gt_depths, opt.pred_out_dir)
    du.save_pred_depths(pred_depths, opt.pred_out_dir, opt.model_name)


# ========== EVALUATION ==========
def eval_depth(opt):
    gt_dir = "ground_truth"
    model_names = os.listdir(opt.pred_out_dir)
    model_names.remove(gt_dir)
    gt_path = os.path.join(opt.pred_out_dir, gt_dir, "depth")
    pred_paths = {model: os.path.join(opt.pred_out_dir, model, "depth") for model in model_names}
    depth_eval_total = pd.DataFrame()

    for model, pred_path in pred_paths.items():
        gt_files = glob.glob(os.path.join(gt_path, "*.npy"))
        gt_files.sort()
        pred_file = os.path.join(pred_path, "kitti_eigen_depth_predictions.npy")
        if not os.path.isfile(pred_file):
            print("No depth prediction file from model {}".format(model))
            continue

        predicted_depths = np.load(pred_file)
        print(model, "predicted shape", predicted_depths.shape)
        result = evaluate_depth_errors(predicted_depths, gt_files, model, opt.min_depth, opt.max_depth)
        depth_eval_total = result if depth_eval_total.empty \
                                   else depth_eval_total.append(result, ignore_index=True)

    print("depth evaluation data:\n", depth_eval_total.head())
    depth_eval_total.to_csv(os.path.join(opt.eval_out_dir, "depth_eval_all.csv"))
    depth_eval_mean = depth_eval_total.groupby(by=["model"]).agg("mean")
    print("depth evaluation result:\n", depth_eval_mean)
    depth_eval_mean.to_csv(os.path.join(opt.eval_out_dir, "depth_eval.csv"))


def evaluate_depth_errors(pred_depths, gt_files, model_name, min_depth, max_depth):
    columns = ["model", "test_frame", "rms", "log_rms", "abs_rel", "sq_rel", "a1", "a2", "a3"]
    depth_eval_result = pd.DataFrame(columns=columns)

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

        abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = \
            du.compute_depth_errors(gt_depth[mask], predi_depth[mask])

        depth_eval_result = depth_eval_result.append(
            dict(zip(columns, [model_name, i, abs_rel, sq_rel, rms, log_rms, a1, a2, a3])),
            ignore_index=True)

    return depth_eval_result


def eval_pose(opt):
    eval_seq = ["09", "10"]
    gt_dir = "ground_truth"
    model_names = os.listdir(opt.pred_out_dir)
    model_names.remove(gt_dir)
    gt_path = os.path.join(opt.pred_out_dir, gt_dir, "pose")
    pred_paths = {model: os.path.join(opt.pred_out_dir, model, "pose") for model in model_names}
    dfcols = ["model", "seq", "subseq_begin", "subseq_index", "trn_err", "rot_err"]
    pose_errors_df = pd.DataFrame(columns=dfcols)

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
                if pred_short_seq.ndim < 2 or abs(gt_short_seq[0, 0] - pred_short_seq[0, 0]) > 0.01:
                    continue

                assert (gt_short_seq.shape == (5, 8) and pred_short_seq.shape[1] == 8)
                try:
                    sseq_err = pu.compute_pose_error(gt_short_seq, pred_short_seq)
                    for pose_err in sseq_err:
                        pose_errors_df = pose_errors_df.append(
                            dict(zip(dfcols, [model, seq, i, pose_err[0], pose_err[1], pose_err[2]])),
                            ignore_index=True)
                except ValueError as ve:
                    print(ve)

    pose_eval_result = evaluate_from_errors(pose_errors_df)
    print(pose_eval_result)
    pose_eval_result.to_csv(os.path.join(opt.eval_out_dir, "pose_eval.csv"))


def evaluate_from_errors(errors_df):
    src_cols = list(errors_df)

    # average translational error of short sequences
    atess = errors_df.groupby(by=src_cols[:3]).agg({"trn_err": "mean"})
    atess = atess.groupby(by=src_cols[:2]).agg({"trn_err": "mean"})
    atess = atess.rename({"trn_err": "atess"})

    # average rotational error of short sequences
    aress = errors_df.groupby(by=src_cols[:3]).agg({"rot_err": "mean"})
    aress = aress.groupby(by=src_cols[:2]).agg({"rot_err": "mean"})
    aress = aress.rename({"rot_err": "aress"})

    grp_cols = [src_cols[i] for i in (0, 1, 3)]
    # average translational error of specific frames in short sequences
    atefr = errors_df.groupby(by=grp_cols).agg({"trn_err": "mean"})
    atefr = atefr.unstack(level=-1)
    atefr_cols = ["te{}f".format(bottom) for top, bottom in atefr.columns.values.tolist()]
    atefr.columns = atefr_cols

    # average rotational error of specific frames in short sequences
    arefr = errors_df.groupby(by=grp_cols).agg({"rot_err": "mean"})
    arefr = arefr.unstack(level=-1)
    arefr_cols = ["re{}f".format(bottom) for top, bottom in arefr.columns.values.tolist()]
    arefr.columns = arefr_cols

    pose_eval_res = pd.concat([atess, aress, atefr, arefr], axis=1)
    return pose_eval_res






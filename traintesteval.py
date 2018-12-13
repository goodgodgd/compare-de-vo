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
from model_operator import GeoNetOperator


# ========== TRAIN ==========
def train(opt, net_model):
    tf.enable_eager_execution()
    set_random_seed()
    print(opt.checkpoint_dir)
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    model_op = GeoNetOperator(opt, net_model) if "geonet" in opt.model_name else None
    print(opt.model_name, model_op)

    def data_feeder():
        return dataset_feeder(opt, "train")
    model_op.train(data_feeder)


def set_random_seed():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ========== PREDICT ==========
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
        print("```````init ckpt", opt.init_ckpt_file)
        saver.restore(sess, opt.init_ckpt_file)
        print("```````````")
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
                    # geonet was trained for inverse pose
                    pred_pose_tum = pu.format_poses_tum(pred_pose_batch[b, :, :],
                                                        gt_pose_batch[b, :, 0], inv=True)
                    pred_poses.append(pred_pose_tum)
                    gt_poses.append(gt_pose_batch[b, :, :])
            except tf.errors.OutOfRangeError:
                print("dataset finished at step", i*opt.batch_size)
                break

    print("output length (gt, pred)", len(gt_poses), len(pred_poses))
    # one can evaluate pose errors here but we save results and evaluate it in the evaluation step
    pu.save_pose_result(pred_poses, frames, opt.pred_out_dir, opt.model_name, opt.seq_length)
    # save_pose_result(gt_poses, opt.pred_out_dir, "ground_truth", frames, opt.seq_length)


def pred_pose_estimator(opt, net_model):
    tf.enable_eager_execution()
    print(opt.pred_out_dir)
    if not os.path.exists(opt.pred_out_dir):
        os.makedirs(opt.pred_out_dir)
    model_op = GeoNetOperator(opt, net_model) if opt.model_name == "geonet" else None
    print("```modelname", opt.model_name)

    # dataset = dataset_feeder(opt, "test")

    def data_feeder():
        return dataset_feeder(opt, "test")

    # prediction result by estimator.predict() returns results of total dataset
    # it is not packed in batch size
    predictions = model_op.predict(data_feeder)

    opt.batch_size = 1
    dataset = dataset_feeder(opt, "test")

    gt_poses = []
    pr_poses = []
    frames = []
    target_ind = (opt.seq_length - 1)//2

    for i, (feat, pred) in enumerate(zip(dataset, predictions)):
        frame = bytes(feat["frame_int8"][0]).decode("utf-8")
        frames.append(frame)
        gt_pose = feat["gt"][0]
        pr_pose = pred["pose"]
        pr_pose = np.insert(pr_pose, target_ind, np.zeros((1, 6)), axis=0)
        # geonet was trained for inverse pose
        pr_pose_tum = pu.format_poses_tum(pr_pose, gt_pose[:, 0], inv=True)
        pr_poses.append(pr_pose_tum)
        gt_poses.append(gt_pose)

    print("output length (gt, pred)", len(gt_poses), len(pr_poses))
    # one can evaluate pose errors here but we save results and evaluate it in the evaluation step
    pu.save_pose_result(gt_poses, frames, opt.pred_out_dir, opt.model_name, opt.seq_length)
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
    eval_drive = ["09", "10"]
    gt_dir = "ground_truth"
    model_names = os.listdir(opt.pred_out_dir)
    model_names.remove(gt_dir)
    gt_path = os.path.join(opt.pred_out_dir, gt_dir, "pose")
    pred_paths = {model: os.path.join(opt.pred_out_dir, model, "pose") for model in model_names}
    sseq_cols = ["model", "drive", "subseq_begin", "subseq_index", "trn_err", "rot_err"]
    subseq_errors = pd.DataFrame(columns=sseq_cols)

    for model, pred_path in pred_paths.items():
        if not os.path.isdir(os.path.join(gt_path, eval_drive[0])):
            print("No pose prediction sequences from model {}".format(model))
            continue

        for drive in eval_drive:
            subseq_errors = evaluate_drive_subseq(gt_path, pred_path, model, drive, subseq_errors)

    pose_eval_result = evaluate_subseq_errors(subseq_errors)
    print(pose_eval_result)
    pose_eval_result.to_csv(os.path.join(opt.eval_out_dir, "pose_eval.csv"))


def evaluate_drive_subseq(gt_path, pred_path, model, drive, subseq_errors):
    sseq_cols = list(subseq_errors)
    gt_files = glob.glob(os.path.join(gt_path, drive, "*.txt"))
    gt_files.sort()

    for i, gtfile in enumerate(gt_files):
        predfile = gtfile.replace(gt_path, pred_path)
        # gt file must exist. if no prediction file, just skip it.
        assert os.path.isfile(gtfile), "{} not found!!".format(gtfile)
        if not os.path.isfile(predfile):
            continue

        # read short (5) pose sequences
        gt_short_seq = np.loadtxt(gtfile)
        pred_short_seq = np.loadtxt(predfile)
        if pred_short_seq.ndim < 2 or abs(gt_short_seq[0, 0] - pred_short_seq[0, 0]) > 0.01:
            continue

        assert (gt_short_seq.shape == (5, 8) and pred_short_seq.shape[1] == 8)
        try:
            # compute error between predicted poses and gt poses
            sseq_err = pu.compute_pose_error(gt_short_seq, pred_short_seq, ambiguous_scale=True)
            for pose_err in sseq_err:
                subseq_errors = subseq_errors.append(
                    dict(zip(sseq_cols, [model, drive, i, pose_err[0], pose_err[1], pose_err[2]])),
                    ignore_index=True)
        except ValueError as ve:
            print(ve)
    return subseq_errors


def evaluate_subseq_errors(errors_df):
    src_cols = list(errors_df)
    # groupby: src_cols[:3] = ["model", "drive", "subseq_begin"]

    # average translational error of short sequences
    atess = errors_df.groupby(by=src_cols[:2]).agg({"trn_err": ["mean", "std"]})
    atess_cols = ["te_{}".format(bottom) for top, bottom in atess.columns.values.tolist()]
    atess.columns = atess_cols

    # average rotational error of short sequences
    aress = errors_df.groupby(by=src_cols[:2]).agg({"rot_err": ["mean", "std"]})
    aress_cols = ["re_{}".format(bottom) for top, bottom in aress.columns.values.tolist()]
    aress.columns = aress_cols

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


def eval_traj(opt):
    eval_drive = ["09", "10"]
    gt_dir = "ground_truth"
    model_names = os.listdir(opt.pred_out_dir)
    model_names.remove(gt_dir)
    gt_path = os.path.join(opt.pred_out_dir, gt_dir, "pose")
    pred_paths = {model: os.path.join(opt.pred_out_dir, model, "pose") for model in model_names}
    traj_cols = ["model", "drive", "interval", "gtind", "trn_err", "rot_err"]
    traj_errors = pd.DataFrame(columns=traj_cols)
    print(pred_paths)

    for model, pred_path in pred_paths.items():
        if not os.path.isdir(os.path.join(gt_path, eval_drive[0])):
            print("No pose prediction sequences from model {}".format(model))
            continue

        for drive in eval_drive:
            # reconstruct full trajectory based on predictd relative poses
            # orb sequences are not fully predicted, so they would not be reconstructed
            if "orb" in model:
                pu.reconstruct_traj_and_save(gt_path, pred_path, drive, 2, True)
            else:
                pu.reconstruct_traj_and_save(gt_path, pred_path, drive, opt.seq_length, False)
            traj_errors = evaluate_drive_traj(gt_path, pred_path, model, drive, traj_errors)

    traj_eval_result = evaluate_traj_errors(traj_errors)
    traj_eval_result.to_csv(os.path.join(opt.eval_out_dir, "traj_eval.csv"))


def evaluate_drive_traj(gt_path, pred_path, model, drive, traj_errors):
    traj_cols = list(traj_errors)
    # gt trajectory file must exists
    gt_file = os.path.join(gt_path, "{}_full.txt".format(drive))
    assert os.path.exists(gt_file)
    gt_traj = pu.read_pose_file(gt_file)

    pred_file_pattern = os.path.join(pred_path, "{}_full_recon*".format(drive))
    pred_files = glob.glob(pred_file_pattern)

    for pred_file in pred_files:
        pred_traj = pu.read_pose_file(pred_file)

        pred_file = os.path.basename(pred_file)
        interval = pred_file.replace(".", "_").split("_")[-2]
        try:
            # compute error between predicted poses and gt poses
            # reconstructed trajectory has physical scale
            traj_err = pu.compute_pose_error(gt_traj, pred_traj, ambiguous_scale=False)
            print("evaluate_drive_traj", model, drive, interval, len(traj_err))
            for pose_err in traj_err:
                traj_errors = traj_errors.append(
                    dict(zip(traj_cols, [model, drive, int(interval), pose_err[0], pose_err[1], pose_err[2]])),
                    ignore_index=True)
        except ValueError as ve:
            print(ve)

    return traj_errors


def evaluate_traj_errors(errors_df):
    src_cols = list(errors_df)
    # groupby: src_cols[:3] = ["model", "drive", "interval"]
    grpkeys = src_cols[:3]

    # average translational error grouped by intervals
    ateitv = errors_df.groupby(by=grpkeys).agg({"trn_err": ["mean", "std"]})
    ateitv_cols = ["te_{}".format(bottom) for top, bottom in ateitv.columns.values.tolist()]
    ateitv.columns = ateitv_cols

    # average rotational error grouped by intervals
    areitv = errors_df.groupby(by=grpkeys).agg({"rot_err": ["mean", "std"]})
    areitv_cols = ["re_{}".format(bottom) for top, bottom in areitv.columns.values.tolist()]
    areitv.columns = areitv_cols

    traj_eval_res = pd.concat([ateitv, areitv], axis=1)
    return traj_eval_res

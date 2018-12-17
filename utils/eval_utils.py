import numpy as np
import utils.pose_utils as pu


# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_pose_seq_error(gt_poses, pred_poses, ambiguous_scale: bool):
    gt_poses, pred_poses, ali_inds = align_pose_seq(gt_poses, pred_poses)

    assert gt_poses.shape == pred_poses.shape, "after alignment, gt:{} == pred:{}"\
        .format(gt_poses.shape, pred_poses.shape)
    seq_len = gt_poses.shape[0]
    err_result = []
    for si in range(1, seq_len):
        te, re = pose_diff(gt_poses[si], pred_poses[si], ambiguous_scale)
        err_result.append([ali_inds[si], te, re])
        assert ali_inds[si] > 0, "{} {}".format(si, ali_inds)
    return err_result


def align_pose_seq(gt_poses, pred_poses, max_diff=0.01):
    # assert np.array_equal(gt_poses[0, 1:], pred_poses[0, 1:]), \
    #     "different initial pose: {}, {}".format(gt_poses[0, 1:], pred_poses[0, 1:])

    gt_times = gt_poses[:, 0].tolist()
    pred_times = pred_poses[:, 0].tolist()
    potential_matches = [(abs(gt - pt), gt, gi, pt, pi)
                         for gi, gt in enumerate(gt_times)
                         for pi, pt in enumerate(pred_times)
                         if abs(gt - pt) < max_diff]
    potential_matches.sort()
    matches = []
    for diff, gt, gi, pt, pi in potential_matches:
        if gt in gt_times and pt in pred_times:
            gt_times.remove(gt)
            pred_times.remove(pt)
            matches.append((gi, pi))
    matches.sort()
    aligned_inds = [gi for gi, pi in matches]

    if len(matches) < 2:
        raise ValueError("aligned poses are {} from {}".format(len(matches), len(potential_matches)))

    aligned_gt = []
    aligned_pred = []
    for gi, pi in matches:
        aligned_gt.append(gt_poses[gi])
        aligned_pred.append(pred_poses[pi])
    aligned_gt = np.array(aligned_gt)
    aligned_pred = np.array(aligned_pred)
    return aligned_gt, aligned_pred, aligned_inds


def pose_diff(gt_pose, pred_pose, ambiguous_scale):
    scale = 1
    if ambiguous_scale:
        if np.sum(pred_pose[1:4] ** 2) > 0.00001:
            # optimize the scaling factor
            scale = np.sum(gt_pose[1:4] * pred_pose[1:4]) / np.sum(pred_pose[1:4] ** 2)
        else:
            "invalid scale division {}, scale=1".format(np.sum(pred_pose[1:4] ** 2))
    # translational error
    alignment_error = pred_pose[1:4] * scale - gt_pose[1:4]
    trn_rmse = np.sqrt(np.sum(alignment_error ** 2))
    # rotation matrices
    gt_rmat = pu.quat2mat(gt_pose[4:])
    pred_rmat = pu.quat2mat(pred_pose[4:])
    # relative rotation
    rel_rmat = np.matmul(np.transpose(gt_rmat), pred_rmat)
    qx, qy, qz, qw = pu.rot2quat(rel_rmat)
    rot_diff = abs(np.arccos(qw))
    return trn_rmse, rot_diff


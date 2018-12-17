import os
import numpy as np
import glob
import utils.pose_utils as pu
import utils.eval_utils as eu

# CAUTION!!
# TUM's pose format use quaternion like (qx qy qz qw)
# one must be careful about quaternion expression


def reconstruct_traj_and_save(gt_path, pred_path, drive, subseq_len, read_full: bool):
    pred_pose_path = os.path.join(pred_path, drive)
    gt_pose_file = os.path.join(gt_path, "{}_full.txt".format(drive))
    assert os.path.isfile(gt_pose_file) and os.path.isdir(pred_pose_path),\
        "files: {}, {}".format(gt_pose_file, pred_pose_path)

    gt_traj = pu.read_pose_file(gt_pose_file)
    traj_len = gt_traj.shape[0]
    prinitv = traj_len//10
    print("gt traj shape", gt_traj.shape)
    print("gt traj\n", gt_traj[0:-1:prinitv, 1:4])

    # when reading full traj, run the for loop only once
    if read_full:
        subseq_len = 2

    # itv: pose multiplication interval
    for itv in range(1, subseq_len):
        if read_full:
            pred_pose_file = os.path.join(pred_path, "{}_full.txt".format(drive))
            if not os.path.exists(pred_pose_file):
                break
            pred_abs_poses = pu.read_pose_file(pred_pose_file)
            pred_rel_poses = create_rel_poses(pred_abs_poses)
        else:
            pred_rel_poses = read_relative_poses(pred_pose_path, itv)

        # align two pose lists by time
        gt_abs_poses, pred_rel_poses, ali_inds = eu.align_pose_seq(gt_traj, pred_rel_poses)
        # convert gt trajectory to relative poses between two adjcent poses
        gt_rel_poses = create_rel_poses(gt_abs_poses)
        assert gt_rel_poses.shape == pred_rel_poses.shape

        recon_traj = reconstruct_abs_poses(pred_rel_poses, gt_rel_poses)
        print("reconstructed trajectory:\n", recon_traj[0:-1:prinitv//itv, 1:4])
        filename = os.path.join(pred_path, "{}_full_recon_{:02d}.txt".format(drive, itv))
        np.savetxt(filename, recon_traj, fmt="%.6f")


def read_relative_poses(pose_path, itv):
    file_list = glob.glob(os.path.join(pose_path, "*.txt"))
    file_list.sort()
    rel_poses = [np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)]
    for i in range(0, len(file_list), itv):
        sub_seq = pu.read_pose_file(file_list[i])
        rel_poses.append(sub_seq[itv])

    rel_poses = np.array(rel_poses)
    return rel_poses


def create_rel_poses(gt_traj):
    gt_rel_poses = [gt_traj[0]]
    for i in range(1, len(gt_traj)):
        base_pose = gt_traj[i - 1]
        cur_pose = gt_traj[i]
        rel_pose = compute_relative_pose(base_pose, cur_pose)
        gt_rel_poses.append(rel_pose)
        # print("create rel poses:{} {}\n    {}\n  {}".format(i, bef_pose, cur_pose, rel_pose))
    gt_rel_poses = np.array(gt_rel_poses)
    return gt_rel_poses


def compute_relative_pose(base_pose, other_pose):
    base_mat = quat_pose_to_mat(base_pose)
    other_mat = quat_pose_to_mat(other_pose)
    rel_mat = np.matmul(np.linalg.inv(base_mat), other_mat)
    rel_quat = pu.rot2quat(rel_mat[:3, :3])
    rel_posi = rel_mat[:3, 3]
    rel_pose = np.concatenate([other_pose[:1], rel_posi, rel_quat], axis=0)
    return rel_pose


def quat_pose_to_mat(pose_vec):
    assert pose_vec.size == 8, "pose to mat {}, {}".format(pose_vec.size, pose_vec.shape)
    Tmat = np.identity(4)
    Tmat[:3, :3] = pu.quat2mat(pose_vec[4:])
    Tmat[:3, 3] = pose_vec[1:4]
    return Tmat


def reconstruct_abs_poses(pred_rel_poses, gt_rel_poses):
    init_pose = compute_relative_pose(pred_rel_poses[0], gt_rel_poses[0])
    pred_traj = [init_pose]
    assert pred_rel_poses.shape == gt_rel_poses.shape
    pose_len = pred_rel_poses.shape[0]

    for i in range(1, pose_len):
        gt_pose = gt_rel_poses[i]
        pred_pose = pred_rel_poses[i]
        if np.sum(pred_pose[1:4] ** 2) > 1e-6:
            scale = np.sum(gt_pose[1:4] * pred_pose[1:4]) / np.sum(pred_pose[1:4] ** 2)
        else:
            scale = 0
        pred_pose[1:4] = pred_pose[1:4] * scale
        pred_abs_pose = multiply_poses(pred_traj[-1], pred_pose)
        # print("multiply:", scale, pred_traj[-1], "\n ", pred_pose, "\n ", pred_abs_pose)
        pred_traj.append(pred_abs_pose)
    pred_traj = np.array(pred_traj)
    return pred_traj


def multiply_poses(base_pose, other_pose):
    assert base_pose.size == 8 and other_pose.size == 8, "relative pose: {}, {}".format(base_pose.size, other_pose.size)
    base_mat = quat_pose_to_mat(base_pose)
    other_mat = quat_pose_to_mat(other_pose)
    abs_mat = np.matmul(base_mat, other_mat)
    abs_quat = pu.rot2quat(abs_mat[:3, :3])
    abs_posi = abs_mat[:3, 3]
    abs_pose = np.concatenate([other_pose[:1], abs_posi, abs_quat], axis=0)
    return abs_pose

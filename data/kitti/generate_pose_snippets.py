import os
import numpy as np
from glob import glob
from data.kitti.pose_evaluation_utils import mat2euler, pose_vec_to_mat, rot2quat


def generate_pose_snippets(dataset_dir, seq_id, seq_length):
    pose_gt_dir = os.path.join(dataset_dir, 'poses')
    seq_dir = os.path.join(dataset_dir, 'sequences', '{:02d}'.format(seq_id))
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    frames = ['%.2d %.6d' % (seq_id, n) for n in range(N)]

    pose_file = os.path.join(pose_gt_dir, '{:02d}.txt'.format(seq_id))
    with open(pose_file, 'r') as f:
        poses = f.readlines()
    poses_gt = []
    for pose in poses:
        pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3,4))
        rot = np.linalg.inv(pose[:,:3])
        tran = -np.dot(rot, pose[:,3].transpose())
        rz, ry, rx = mat2euler(rot)
        poses_gt.append(tran.tolist() + [rx, ry, rz])
    poses_gt = np.array(poses_gt)

    half_offset = (seq_length - 1)//2
    pose_sequences = []
    for tgt_idx in range(N):
        # pad invalid range
        if not is_valid_sample(frames, tgt_idx, seq_length):
            pose_sequences.append(0)
            continue
        # add short pose sequence
        pred_poses = poses_gt[tgt_idx - half_offset:tgt_idx + half_offset + 1]
        pose_seq = pose_seq_TUM(pred_poses)
        pose_sequences.append(pose_seq)
    return pose_sequences


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    assert (tgt_drive == min_src_drive and tgt_drive == max_src_drive)
    return True


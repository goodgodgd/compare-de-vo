import os
import numpy as np
from glob import glob
from data.kitti.pose_evaluation_utils import mat2euler, dump_pose_seq_TUM


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
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False


def generate_pose_snippets(opt, seq_id):
    pose_gt_dir = os.path.join(opt.dataset_dir, 'poses')
    seq_dir = os.path.join(opt.dataset_dir, 'sequences', '{:02d}'.format(seq_id))
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (seq_id, n) for n in range(N)]
    with open(os.path.join(seq_dir, 'times.txt'), 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

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

    out_dir = os.path.join(opt.dump_root, '{:02d}'.format(seq_id))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    max_src_offset = (opt.seq_length - 1)//2
    for tgt_idx in range(N):
        if not is_valid_sample(test_frames, tgt_idx, opt.seq_length):
            continue
        pred_poses = poses_gt[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        out_file = os.path.join(out_dir, '{:06d}_pose.txt'.format(tgt_idx))
        dump_pose_seq_TUM(out_file, pred_poses, curr_times)

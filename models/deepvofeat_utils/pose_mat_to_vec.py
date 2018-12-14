import numpy as np
import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if module_path not in sys.path: sys.path.append(module_path)
from data.kitti.kitti_pose_utils import mat2euler, format_poses_tum, read_pose_file


predicts_path = "/home/ian/workplace/DevoBench/devo_bench_data/predicts"
gt_path = predicts_path + "/ground_truth/pose"
pred_path = predicts_path + "/deepvofeat/pose"
drives = ["09", "10"]
assert os.path.isdir(predicts_path)


def main():
    for drive in drives:
        # read files
        gt_pose_file = os.path.join(gt_path, "{}_full.txt".format(drive))
        pred_pose_src = os.path.join(pred_path, "{}.txt".format(drive))
        pred_pose_dst = os.path.join(pred_path, "{}_full.txt".format(drive))

        gt_poses_tum = read_pose_file(gt_pose_file)
        pred_poses_kit = np.loadtxt(pred_pose_src)
        print("shapes", gt_poses_tum.shape, pred_poses_kit.shape)
        pred_poses_xyz = pred_poses_kit[:, [3, 7, 11]]

        # convert kitti format poses to trans + euler angles
        pred_poses = []
        for pose in pred_poses_kit:
            pose = np.reshape(pose, (3, 4))
            rot = pose[:, :3]   # np.linalg.inv(pose[:, :3])
            tran = pose[:, 3].transpose()   # -np.dot(rot, pose[:, 3].transpose())
            rz, ry, rx = mat2euler(rot)
            pred_poses.append(tran.tolist() + [rx, ry, rz])
        pred_poses = np.array(pred_poses)
        np.savetxt(os.path.join(pred_path, "{}_euler.txt".format(drive)), pred_poses, fmt="%.06f")

        # format to tum and save full trajectory
        pred_poses_tum = format_poses_tum(pred_poses, gt_poses_tum[:, 0])
        np.savetxt(pred_pose_dst, pred_poses_tum, fmt="%.06f")
        print("pred tum", pred_poses_tum.shape)

        # slice trajectory into 5 frame sequences and save them
        if not os.path.isdir(pred_path + "/" + drive):
            os.makedirs(pred_path + "/" + drive)
        full_seq_length = pred_poses_tum.shape[0]
        seq_length = 5
        for stt_idx in range(full_seq_length - seq_length + 1):
            pose_sseq = pred_poses[stt_idx: stt_idx+seq_length]
            time_seq = gt_poses_tum[stt_idx: stt_idx+seq_length, 0]
            pose_sseq = format_poses_tum(pose_sseq, time_seq)
            dump_file = os.path.join(pred_path, drive, "{:06d}.txt".format(stt_idx))
            np.savetxt(dump_file, pose_sseq, fmt='%.6f')


if __name__ == '__main__':
    main()

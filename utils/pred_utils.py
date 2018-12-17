import os
import numpy as np


def save_pose_result(pose_seqs, frames, output_root, modelname, seq_length):
    assert os.path.isdir(output_root), "[ERROR] dir not found: {}".format(output_root)
    save_path = os.path.join(output_root, modelname, "pose")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    sequences = []
    for i, (poseseq, frame) in enumerate(zip(pose_seqs, frames)):
        seq_id, frame_id = frame.split(" ")
        if seq_id not in sequences:
            sequences.append(seq_id)
            if not os.path.isdir(os.path.join(save_path, seq_id)):
                os.makedirs(os.path.join(save_path, seq_id))

        half_seq = (seq_length - 1) // 2
        filename = os.path.join(save_path, seq_id, "{:06d}.txt".format(int(frame_id)-half_seq))
        np.savetxt(filename, poseseq, fmt="%06f")
    print("pose results were saved!! at", save_path)


def save_gt_depths(depths, output_root):
    if not os.path.isdir(output_root):
        raise FileNotFoundError(output_root)

    save_path = os.path.join(output_root, "ground_truth", "depth")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i, depth in enumerate(depths):
        filename = os.path.join(save_path, "{:06d}".format(i))
        np.save(filename, depth)


def save_pred_depths(depths, output_root, modelname):
    if not os.path.isdir(os.path.join(output_root, modelname)):
        raise FileNotFoundError()

    save_path = os.path.join(output_root, modelname, "depth")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    depths = np.concatenate(depths, axis=0)
    filename = os.path.join(save_path, "kitti_eigen_depth_predictions")
    np.save(filename, depths)
    print("predicted depths were saved!! shape=", depths.shape)



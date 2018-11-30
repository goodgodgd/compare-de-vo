# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_odom_loader.py
import sys
import numpy as np
from glob import glob
import os
import scipy.misc

module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if module_path not in sys.path: sys.path.append(module_path)
from abstracts import DataLoader
from data.kitti.pose_evaluation_utils import mat2euler, format_pose_seq_TUM


class KittiOdomLoader(DataLoader):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]
        self.half_offset = int((self.seq_length - 1) / 2)

        self.collect_frames()

    def collect_frames(self):
        for seq in self.train_seqs:
            frames = self.collect_sequence_frames(seq)
            self.train_frames.extend(frames)
            gts = self.generate_pose_snippets(self.dataset_dir, seq, self.seq_length)
            self.train_gts.extend(gts)
            self.intrinsics[seq] = self.load_intrinsics('{:02d}'.format(seq), 0)
        self.num_train = len(self.train_frames)
        assert self.num_train == len(self.train_gts), \
            "num train: {} {}".format(self.num_train, len(self.train_gts))

        for seq in self.test_seqs:
            frames = self.collect_sequence_frames(seq)
            self.test_frames.extend(frames)
            gts = self.generate_pose_snippets(self.dataset_dir, seq, self.seq_length)
            self.test_gts.extend(gts)
            self.intrinsics[seq] = self.load_intrinsics('{:02d}'.format(seq), 0)
        self.num_test = len(self.test_frames)
        assert self.num_test == len(self.test_gts), \
            "num train: {} {}".format(self.num_test, len(self.test_gts))

    def collect_sequence_frames(self, seq):
        split_frames = []
        seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
        img_dir = os.path.join(seq_dir, 'image_2')
        N = len(glob(img_dir + '/*.png'))
        for n in range(N):
            split_frames.append('%.2d %.6d' % (seq, n))
        return split_frames

    def generate_pose_snippets(self, dataset_dir, seq_id, seq_length):
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
            pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3, 4))
            rot = np.linalg.inv(pose[:, :3])
            tran = -np.dot(rot, pose[:, 3].transpose())
            rz, ry, rx = mat2euler(rot)
            poses_gt.append(tran.tolist() + [rx, ry, rz])
        poses_gt = np.array(poses_gt)

        half_offset = (seq_length - 1) // 2
        pose_sequences = []
        for tgt_idx in range(N):
            # pad invalid range
            if not self.is_valid_sample(frames, tgt_idx):
                pose_sequences.append(0)
                continue
            # add short pose sequence
            pred_poses = poses_gt[tgt_idx - half_offset:tgt_idx + half_offset + 1]
            pose_seq = format_pose_seq_TUM(pred_poses)
            pose_sequences.append(pose_seq)
        return pose_sequences

    # ========================================
    # after initialized, it feeds example one by one

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def get_test_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.test_frames, tgt_idx):
            return False
        example = self.load_example(self.test_frames, tgt_idx)
        return example

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        min_src_idx = tgt_idx - self.half_offset
        max_src_idx = tgt_idx + self.half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        example = dict()
        example['image_seq'] = image_seq
        example['intrinsics'] = self.scale_intrinsics(self.intrinsics[int(tgt_drive)], zoom_x, zoom_y)
        example['gt'] = self.train_gts[tgt_idx]
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        return example

    def load_image_sequence(self, frames, tgt_idx):
        image_seq = []
        for o in range(-self.half_offset, self.half_offset+1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, 'sequences', '%s/image_2/%s.png' % (drive, frame_id))
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir, 'sequences', '%s/calib.txt' % drive)
        proj_c2p, _ = self.read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics
    
    @staticmethod
    def read_calib_file(filepath, cid=2):
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            calib = f.readlines()
        
        def parse_line(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data

        proj_c2p = parse_line(calib[cid], shape=(3,4))
        proj_v2c = parse_line(calib[-1], shape=(3,4))
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c

    @staticmethod
    def scale_intrinsics(mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out

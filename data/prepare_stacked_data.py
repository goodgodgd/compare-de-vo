# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py
import os
import sys
import argparse
import scipy.misc
import numpy as np
from glob import glob

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path: sys.path.append(module_path)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--split",         type=str, default="all", choices=["all", "train", "test"])
parser.add_argument("--img_height",    type=int, default=128,   help="image height")
parser.add_argument("--img_width",     type=int, default=416,   help="image width")
parser.add_argument("--num_threads",   type=int, default=4,     help="number of threads to use")
parser.add_argument("--remove_static", help="remove static frames from kitti raw data", action='store_true')
opt = parser.parse_args()

dump_list = []


def concat_image_seq(seq):
    # when single image, return it right away
    if isinstance(seq, np.ndarray):
        return seq
    # seq: list of numpy images
    res = None
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n, data_feeder, num_split):
    print_progress(n, num_split)
    example = data_feeder(n)
    if example is False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    gt = example['gt']
    intrinsics = example['intrinsics']
    dump_dir = os.path.join(opt.dump_root, example['folder_name'])

    # keep intrinsics for each dir
    if dump_dir not in dump_list:
        dump_list.append(dump_dir)
        dump_cam_file = '{}/intrinsics.txt'.format(dump_dir)
        np.savetxt(dump_cam_file, intrinsics, fmt='%.6f')

    try:
        os.makedirs(os.path.join(dump_dir, 'gt'))
    except OSError:
        if not os.path.isdir(os.path.join(dump_dir, 'gt')):
            raise

    # save stacked image
    # dump_img_file = '{}/{}.jpg'.format(dump_dir, example['file_name'])
    # scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))

    # save gt data
    if "odom" in opt.dataset_name:
        dump_gt_file = '{}/gt/{}_gt.txt'.format(dump_dir, example['file_name'])
        np.savetxt(dump_gt_file, gt, fmt='%.6f')
    elif "eigen" in opt.dataset_name and gt is not None:
        dump_gt_file = '{}/gt/{}_gt.npy'.format(dump_dir, example['file_name'])
        np.save(dump_gt_file, gt)


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total
    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {:d}/{:d}, {:.1%}".format(count, total, pct_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def write_train_frames():
    np.random.seed(8964)
    subfolders = os.listdir(opt.dump_root)
    with open(os.path.join(opt.dump_root, 'train.txt'), 'w') as tf:
        with open(os.path.join(opt.dump_root, 'val.txt'), 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(opt.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(opt.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))


def write_frames_two_splits(check_validity, frames, filename):
    with open(os.path.join(opt.dump_root, filename), 'w') as f:
        for i, frame in enumerate(frames):
            if check_validity(frames, i):
                f.write(frame+'\n')


def write_frames_three_splits(frames, filename):
    with open(os.path.join(opt.dump_root, filename), 'w') as f:
        for i, frame in enumerate(frames):
            drive, camid, frameid = frame.split(' ')
            f.write("{}_{} {}\n".format(drive, camid, frameid))


def main():
    if not os.path.exists(opt.dump_root):
        os.makedirs(opt.dump_root)

    data_loader = None

    if opt.dataset_name == 'kitti_odom':
        from data.kitti.kitti_odom_loader import KittiOdomLoader
        data_loader = KittiOdomLoader(opt.dataset_dir,
                                      img_height=opt.img_height,
                                      img_width=opt.img_width,
                                      seq_length=opt.seq_length)

    if opt.dataset_name == 'kitti_raw_eigen':
        from data.kitti.kitti_raw_loader import KittiRawLoader
        data_loader = KittiRawLoader(opt.dataset_dir,
                                     split='eigen',
                                     img_height=opt.img_height,
                                     img_width=opt.img_width,
                                     seq_length=opt.seq_length,
                                     remove_static=opt.remove_static)

    if opt.dataset_name == 'kitti_raw_stereo':
        from data.kitti.kitti_raw_loader import KittiRawLoader
        data_loader = KittiRawLoader(opt.dataset_dir,
                                     split='stereo',
                                     img_height=opt.img_height,
                                     img_width=opt.img_width,
                                     seq_length=opt.seq_length,
                                     remove_static=opt.remove_static)

    if opt.dataset_name == 'cityscapes':
        from data.cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(opt.dataset_dir,
                                        img_height=opt.img_height,
                                        img_width=opt.img_width,
                                        seq_length=opt.seq_length)

    if opt.split in ["all", "train"]:
        def train_feeder(n):
            return data_loader.get_train_example_with_idx(n)
        # save train/val data
        for n in range(data_loader.num_train):
            dump_example(n, train_feeder, data_loader.num_train)
        # save train/val file list in the exactly same way with GeoNet
        write_train_frames()
        print("\nfinished writing train frames!!")

    # IMPORTANT! train/val data MUST be processed before test

    if opt.split in ["all", "test"]:
        def test_feeder(n):
            return data_loader.get_test_example_with_idx(n)
        # save test data
        for n in range(data_loader.num_test):
            dump_example(n, test_feeder, data_loader.num_test)

        # save test file list
        def is_valid_sample(frames, idx):
            return data_loader.is_valid_sample(frames, idx)
        if opt.dataset_name == 'kitti_odom':
            write_frames_two_splits(is_valid_sample, data_loader.test_frames, "test.txt")
        if opt.dataset_name == 'kitti_raw_eigen':
            write_frames_three_splits(data_loader.test_frames, "test.txt")
        print("\nfinished writing test frames!!")


if __name__ == '__main__':
    main()

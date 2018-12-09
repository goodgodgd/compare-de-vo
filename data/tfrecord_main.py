import os
import sys
module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path: sys.path.append(module_path)
from data.tfrecord_maker import KittiTfrdMaker
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_odom", "kitti_raw_eigen", "kitti_raw_stereo", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, default=0,     help="sequence length")

opt = parser.parse_args()


def make_to_tfrecords():
    print("dataset_dir={}\ndump_root={}".format(opt.dataset_dir, opt.dump_root))
    tfmaker = KittiTfrdMaker(opt)
    tfmaker.make('train')
    tfmaker.make('val')
    tfmaker.make('test')


if __name__ == "__main__":
    make_to_tfrecords()

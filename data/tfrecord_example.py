from tfrecord_maker import KittiOdomTfrdMaker
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--img_height",    type=int, default=0,     help="image height")
parser.add_argument("--img_width",     type=int, default=0,     help="image width")
opt = parser.parse_args()


def convert_to_tfrecords():
    mnist_cvt = KittiOdomTfrdMaker(opt)
    mnist_cvt.convert('train')
    mnist_cvt.convert('val')
    mnist_cvt.convert('test')


if __name__ == "__main__":
    convert_to_tfrecords()

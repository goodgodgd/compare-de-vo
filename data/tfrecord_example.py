from tfrecord_maker import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--img_height",    type=int, default=128,   help="image height")
parser.add_argument("--img_width",     type=int, default=416,   help="image width")
opt = parser.parse_args()


def mnist_multi_label():
    mnist_cvt = MnistMultiLabelConverter(opt)

    # check data samples
    images = mnist_cvt.datafeeders["train"]["image"].get_slice(list(range(9)))
    targets = mnist_cvt.datafeeders["train"]["target"].get_slice(list(range(9)))
    labels = mnist_cvt.datafeeders["train"]["label"].get_slice(list(range(9)))
    print("image", images[:, :, 4])
    print("target", targets)
    print("label", labels)

    tfrecord_path = pp.processed_data_path("mnist/mnist_ml_tfrecord", ensure_path_exist=True)
    mnist_cvt.convert(tfrecord_path, 'mnist_ml', 'train')
    mnist_cvt.convert(tfrecord_path, 'mnist_ml', 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("data_type")
    parser.add_argument("--datatype", help="data types: mnist_single, mnist_multi, stock_multi")
    parser.set_defaults(datatype="stock_multi")
    options = parser.parse_args()

    mnist_multi_label()


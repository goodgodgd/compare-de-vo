import os
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2


module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path: sys.path.append(module_path)
from models.tfrecord_feeder import dataset_feeder


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='viridis'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


def gray2rgb(im, cmap):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def resize_tensor(image, dshape):
    # image = image.numpy()
    return cv2.resize(image, dsize=(dshape[1], dshape[0]))


def main():
    tf.enable_eager_execution()

    # set dataset feeder options
    opt = namedtuple("options", "tfrecords_dir batch_size seq_length num_source num_scales "
                                "img_height img_width train_epochs")
    opt.tfrecords_dir = "/home/ian/workplace/DevoBench/devo_bench_data/tfrecords/kitti_raw_eigen"
    opt.batch_size = 1
    opt.seq_length = 1
    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4
    opt.img_height = 128
    opt.img_width = 416
    opt.train_epochs = 1

    # read images and gt depths
    images = []
    dataset = dataset_feeder(opt, "test")
    if tf.executing_eagerly():
        for i, features in enumerate(dataset):
            images.append(features["target"][0].numpy())
    images = np.array(images)

    # read predicted depths
    gt_dir = "ground_truth"
    pred_out_dir = "/home/ian/workplace/DevoBench/devo_bench_data/predicts"
    model_names = os.listdir(pred_out_dir)
    model_names.remove(gt_dir)
    pred_paths = {model: os.path.join(pred_out_dir, model, "depth") for model in model_names}
    pred_depths_all = dict()

    for model, pred_path in pred_paths.items():
        # read predicted depth file
        pred_file = os.path.join(pred_path, "kitti_eigen_depth_predictions.npy")
        if not os.path.isfile(pred_file):
            continue

        pred_depths = np.load(pred_file)
        if pred_depths.shape[2] != 416:
            depths_resize = []
            for depth in pred_depths:
                depth = resize_tensor(depth, (128, 416))
                depths_resize.append(depth)
            pred_depths = np.array(depths_resize)
        pred_depths_all[model] = pred_depths

    frame = 500
    plt.figure(figsize=(10, 10))
    plt.subplot(5, 1, 1)
    plt.imshow(images[frame, :, :])
    plt.subplot(5, 1, 2)
    plt.imshow(normalize_depth_for_display(pred_depths_all["sfmlearner_data"][frame, :, :]))
    plt.subplot(5, 1, 3)
    plt.imshow(normalize_depth_for_display(pred_depths_all["deepvofeat"][frame, :, :]))
    plt.subplot(5, 1, 4)
    plt.imshow(normalize_depth_for_display(pred_depths_all["geonet"][frame, :, :]))
    plt.show()


if __name__ == '__main__':
    main()

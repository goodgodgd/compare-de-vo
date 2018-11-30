import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import csv
from datafeeders_for_tfrd import *


class TfrecordsMaker:
    def __init__(self, _opt):
        self.opt = _opt

    def _load_split(self, split):
        raise NotImplementedError()
    
    def make(self, split):
        print("\n==================================================")
        opt = self.opt
        data_feeders = self._load_split(split)

        num_images = len(data_feeders["image"])
        num_shards = max(min(num_images // 5000, 10), 1)
        num_images_per_shard = num_images // num_shards
        print("tfrecord maker started: dataset_name={}, split={}".format(opt.dataset_name, split))
        print("num images, shards, images per shard", num_images, num_shards, num_images_per_shard)

        for si in range(num_shards):
            outfile = "{}/{}_{}_{:04d}.tfrecord".format(opt.dump_root, opt.dataset_name, split, si)
            print("\nconverting to:", outfile)

            with tf.python_io.TFRecordWriter(outfile) as writer:
                for mi in range(num_images_per_shard):
                    di = si*num_images_per_shard + mi
                    if di >= num_images:
                        break

                    # print the percentage-progress.
                    self.print_progress(count=di, total=num_images - 1)

                    raw_example = self.create_next_example_dict(data_feeders)
                    serialized = self.make_serialized_example(raw_example)
                    writer.write(serialized)

        print("\ntfrecord maker finished: dataset_name={}, split={}".format(opt.dataset_name, split))

    @staticmethod
    def create_next_example_dict(feeders):
        example = dict()
        for key, datafeeder in feeders.items():
            example[key] = datafeeder.get_next()
        return example

    @staticmethod
    def make_serialized_example(data_dict):
        # wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data_dict)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

    @staticmethod
    def print_progress(count, total):
        # Percentage completion.
        pct_complete = float(count) / total
        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {:d}/{:d}, {:.1%}".format(count, total, pct_complete)
        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()


class KittiTfrdMaker(TfrecordsMaker):
    def __init__(self, srcpath):
        super().__init__(srcpath)

    def _load_split(self, split):
        # TODO: 바뀐 입력데이터에 맞춰 수정할 것

        dataset_dir = self.opt.dataset_dir
        dstsize = (self.opt.img_width, self.opt.img_height)
        list_file = "{}/{}.txt".format(dataset_dir, split)
        print("list file", list_file)
        image_list, intrin_list = self._read_filelist(dataset_dir, list_file)
        assert len(image_list) == len(intrin_list)

        def resize(image, dsize=dstsize):
            width, height = dsize
            if width > 0 and height > 0:
                return cv2.resize(image, dsize)
            else:
                return image

        def reshape(intrinsic):
            return np.reshape(intrinsic, (3, 3))

        data_feeders = {
            'image': FileFeeder(image_list, _preproc_fn=resize),
            'intrinsic': ConstFeeder(intrinsics, num_frames),
            'gt': ListFeeder(gt_list, _preproc_fn=reshape),
        }
        return data_feeders

    @staticmethod
    def _read_filelist(dataset_dir, list_file):
        image_files = []
        intrin_files = []
        with open(list_file, 'r') as f:
            for line in f:
                paths = line.split(" ")
                filepath = os.path.join(dataset_dir, paths[0], paths[1])
                imagefile = filepath[:-1] + ".jpg"
                intrinfile = filepath[:-1] + "_cam.txt"
                image_files.append(imagefile)
                intrin_files.append(intrinfile)
        return image_files, intrin_files

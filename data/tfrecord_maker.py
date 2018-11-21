import os
import settings
import sys
import glob
import math
import tensorflow as tf
import numpy as np
from utils.constants import PricePredictionCode, PredictionTime
from enum import Enum
import cv2
import csv


class DataFormat(Enum):
    IMAGE = 1
    CSV = 2
    INT64 = 3


class RawDataFeeder:
    def __init__(self, dataset_dir, list_file, data_format: DataFormat, _preproc_fn):
        self.files = self._read_filelist(dataset_dir, list_file)
        self._preproc_fn = _preproc_fn
        self.format = data_format
        self.idx = 0

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read_filelist(dataset_dir, list_file):
        files = []
        with open(list_file, 'rb') as f:
            for line in f:
                paths = line.split(" ")
                file = os.path.join(dataset_dir, paths[0], paths[1])
                files.append(file)
        return files

    # -> get next
    def get_next(self):
        self.idx = self.idx + 1
        onedata = None
        if self.format == DataFormat.IMAGE:
            image = cv2.imread(self.files[self.idx])
            onedata = image
            onedata = onedata.astype(np.uint8)
        elif self.format == DataFormat.CSV:
            with open(self.files[self.idx], 'rb') as f:
                reader = csv.reader(f)
                intrinsics = list(reader)
                intrinsics = np.array(intrinsics)
                onedata = np.reshape(intrinsics, (3, 3))
                onedata = onedata.astype(np.float32)

        if onedata is None:
            raise ValueError("empty data in RawDataFeeder")
        
        # wrap a single raw data as tf.train.Features() and return it 
        return self.convert_to_feature(onedata)
    
    def convert_to_feature(self, rawdata):
        # TODO: 모든 데이터에 대해서 if 검사하지 말고 조건에 따라 다른 함수를 생성하자
        if self.format == DataFormat.IMAGE or self.format == DataFormat.CSV:
            bytes_data = rawdata.tostring()
            return self.wrap_bytes(bytes_data)
        elif self.format == DataFormat.INT64:
            int64_data = rawdata.astype(np.int64)
            return self.wrap_int64(int64_data)
    
    @staticmethod
    def wrap_bytes(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def wrap_int64(value):
        if isinstance(value, np.ndarray):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TfrecordConverter:
    def __init__(self, srcpath):
        self.datafeeders = dict()
        self._load_data(srcpath)

    def _load_data(self, srcpath):
        self._load_split(srcpath, "train")
        self._load_split(srcpath, "val")

    def _load_split(self, srcpath, split):
        raise NotImplementedError()
    
    def convert(self, tfrecords_path, dataset_name, split):
        split_data = self.datafeeders[split]
        num_images = len(split_data["label"])
        num_shards = max(min(num_images // 100000, 10), 1)
        num_images_per_shard = num_images // num_shards
        print("\ntfrecord maker started: dataset_name={}, split={}".format(dataset_name, split))
        print("num images, shards, images per shard", num_images, num_shards, num_images_per_shard)

        for si in range(num_shards):
            outfile = "{}/{}_{}_{:04d}.tfrecord".format(tfrecords_path, dataset_name, split, si)
            print("\bConverting: " + outfile)

            with tf.python_io.TFRecordWriter(outfile) as writer:
                for mi in range(num_images_per_shard):
                    di = si*num_images_per_shard + mi
                    if di >= num_images:
                        break

                    # print the percentage-progress.
                    self.print_progress(count=di, total=num_images - 1)

                    raw_example = self.create_next_example_dict(split_data)
                    serialized = self.make_serialized_example(raw_example)
                    writer.write(serialized)

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
        msg = "\r- Progress: {0:.1%}".format(pct_complete)
        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()


class MnistMultiLabelConverter(TfrecordConverter):
    def __init__(self, srcpath):
        super().__init__(srcpath)

    def _load_split(self, srcpath, split):
        # mnist image pattern
        image_pattern = "{}/{}_data.npy".format(srcpath, split)
        # mnist label pattern
        target_pattern = "{}/{}_label.npy".format(srcpath, split)
        label_pattern = "{}/{}_label.npy".format(srcpath, split)

        def preproc_image(image):
            return image.astype(np.float32)

        def preproc_target(data):
            np.random.seed(0)
            return np.random.randint(4, size=(data.shape[0]))

        def preproc_label(data):
            # labeling table is create by random but result is fixed by 'seed'
            np.random.seed(0)
            tablesize = [10, 4]
            table = np.random.randint(PricePredictionCode.PRICE_FALL, PricePredictionCode.PRICE_RISE + 1, tablesize)
            print("label table (rows for digits, cols for time):\n", table)
            # convert single label to multi label
            multi_label = table[data, :]
            return multi_label.astype(np.uint8)

        self.datafeeders[split] = {
            'image': NpyDataFeeder(dataset_dir, list_file, data_format, _preproc_fn),
        }

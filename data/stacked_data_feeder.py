from enum import Enum
import numpy as np
import tensorflow as tf
import cv2


class DataFormat(Enum):
    IMAGE = 1
    CSV = 2
    INT64 = 3


class RawDataFeeder(object):
    def __len__(self):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()

    def convert_to_feature(self, rawdata):
        raise NotImplementedError()

    @staticmethod
    def wrap_bytes(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def wrap_int64(value):
        if isinstance(value, np.ndarray):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class FileFeeder(RawDataFeeder):
    def __init__(self, name, file_list, dtype, shape_out, preproc_fn=None):
        self.name = name
        self.files = file_list
        self.dtype = dtype
        self.preproc_fn = preproc_fn
        self.shape_out = shape_out
        self.idx = -1

    def __len__(self):
        return len(self.files)

    def get_next(self):
        self.idx = self.idx + 1
        if self.idx >= len(self.files):
            raise IndexError()

        onedata = self.read_file(self.files[self.idx])
        if onedata is None:
            raise FileNotFoundError(self.files[self.idx])

        onedata = onedata.astype(self.dtype)

        if self.preproc_fn is not None:
            onedata = self.preproc_fn(onedata)

        # wrap a single raw data as tf.train.Features()
        features = dict()
        features[self.name] = self.convert_to_feature(onedata)
        if self.shape_out:
            shape = np.array(onedata.shape, dtype=np.int32)
            features[self.name+'_shape'] = self.convert_to_feature(shape)
        return features

    @staticmethod
    def read_file(filename):
        raise NotImplementedError()

    def convert_to_feature(self, rawdata):
        bytes_data = rawdata.tostring()
        return self.wrap_bytes(bytes_data)


class ImageFeeder(FileFeeder):
    def __init__(self, name, file_list, dtype, shape_out, preproc_fn=None):
        super().__init__(name, file_list, dtype, shape_out, preproc_fn)
        print("ImageFeeder created for {} files".format(len(file_list)))

    @staticmethod
    def read_file(filename):
        image = cv2.imread(filename)
        return image


class TextFeeder(FileFeeder):
    def __init__(self, name, file_list, dtype, shape_out, preproc_fn=None):
        super().__init__(name, file_list, dtype, shape_out, preproc_fn)
        print("TextFeeder created for {} files".format(len(file_list)))

    @staticmethod
    def read_file(filename):
        text = np.loadtxt(filename, delimiter=',')
        return text


class NpyFeeder(FileFeeder):
    def __init__(self, name, file_list, dtype, shape_out, preproc_fn=None):
        super().__init__(name, file_list, dtype, shape_out, preproc_fn)
        print("NpyFeeder created for {} files".format(len(file_list)))

    @staticmethod
    def read_file(filename):
        data = np.load(filename)
        return data


class ConstFeeder(RawDataFeeder):
    def __init__(self, name, data, length, shape_out):
        self.name = name
        self.data = data
        self.length = length
        self.shape_out = shape_out
        print("ConstFeeder created, len={}".format(length))

    def __len__(self):
        return self.length

    def get_next(self):
        # wrap a single raw data as tf.train.Features()
        features = dict()
        features[self.name] = self.convert_to_feature(self.data)
        if self.shape_out:
            shape = np.array(self.data.shape, dtype=np.int32)
            features[self.name+'_shape'] = self.convert_to_feature(shape)
        return features

    def convert_to_feature(self, rawdata):
        bytes_data = rawdata.tostring()
        return self.wrap_bytes(bytes_data)


class IntegerFeeder(RawDataFeeder):
    def __init__(self, name, value, length):
        self.name = name
        self.value = value
        self.length = length
        print("IntegerFeeder created, len={}".format(length))

    def __len__(self):
        return self.length

    def get_next(self):
        # wrap a single raw data as tf.train.Features()
        return {self.name: self.convert_to_feature(self.value)}

    def convert_to_feature(self, value):
        return self.wrap_int64(value)


class StringFeeder(RawDataFeeder):
    def __init__(self, name, strings):
        self.name = name
        self.strings = strings
        self.idx = -1
        print("StringFeeder created, len={}".format(len(strings)))

    def __len__(self):
        return len(self.strings)

    def get_next(self):
        self.idx = self.idx + 1
        if self.idx >= len(self.strings):
            raise IndexError()
        # wrap a single raw data as tf.train.Features()
        return {self.name: self.convert_to_feature(self.strings[self.idx])}

    def convert_to_feature(self, string):
        return self.wrap_bytes(bytes(string, 'utf-8'))

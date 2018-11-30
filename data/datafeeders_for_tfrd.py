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


class ImageFeeder(RawDataFeeder):
    def __init__(self, file_list, dtype, preproc_fn=None):
        self.files = file_list
        self.dtype = dtype
        self.preproc_fn = preproc_fn
        self.idx = -1
        print("FileFeeder created for {} files".format(len(file_list)))

    def __len__(self):
        return len(self.files)

    def get_next(self):
        self.idx = self.idx + 1
        image = cv2.imread(self.files[self.idx])
        onedata = image.astype(self.dtype)
        if onedata is None:
            raise ValueError("File Not exist: {}".format(self.files[self.idx]))

        if self.preproc_fn is not None:
            onedata = self.preproc_fn(onedata)
        # wrap a single raw data as tf.train.Features()
        return self.convert_to_feature(onedata)

    def convert_to_feature(self, rawdata):
        bytes_data = rawdata.tostring()
        return self.wrap_bytes(bytes_data)


class TextFeeder(RawDataFeeder):
    def __init__(self, file_list, dtype, preproc_fn=None):
        self.files = file_list
        self.dtype = dtype
        self.preproc_fn = preproc_fn
        self.idx = -1
        print("FileFeeder created for {} files".format(len(file_list)))

    def __len__(self):
        return len(self.files)

    def get_next(self):
        self.idx = self.idx + 1
        data = np.loadtxt(self.files[self.idx])
        onedata = data.astype(self.dtype)
        if onedata is None:
            raise ValueError("File Not exist: {}".format(self.files[self.idx]))

        if self.preproc_fn is not None:
            onedata = self.preproc_fn(onedata)
        # wrap a single raw data as tf.train.Features()
        return self.convert_to_feature(onedata)

    def convert_to_feature(self, rawdata):
        bytes_data = rawdata.tostring()
        return self.wrap_bytes(bytes_data)


# mainly for intrinsics
class ConstFeeder(RawDataFeeder):
    def __init__(self, data, length):
        self.data = data
        self.length = length
        print("ConstFeeder created, len={}".format(length))

    def __len__(self):
        return self.length

    def get_next(self):
        # wrap a single raw data as tf.train.Features()
        return self.convert_to_feature(self.data)

    def convert_to_feature(self, rawdata):
        bytes_data = rawdata.tostring()
        return self.wrap_bytes(bytes_data)





# def convert_to_feature(self, rawdata):
#     # vTODO: 모든 데이터에 대해서 if 검사하지 말고 조건에 따라 다른 함수를 생성하자
#     if self.format == DataFormat.IMAGE or self.format == DataFormat.CSV:
#         bytes_data = rawdata.tostring()
#         return self.wrap_bytes(bytes_data)
#     elif self.format == DataFormat.INT64:
#         int64_data = rawdata.astype(np.int64)
#         return self.wrap_int64(int64_data)

import os.path
import sys
import math
import tensorflow as tf
import numpy as np


# tfrecord file format allows you handle large datasets 'out of memory'.
# in addition, you can contain different data formats in a single file
# and easily apply batch, shuffle on tfrecord files

class TfrecordMaker:
    def __init__(self, srcpath):
        self.data = dict()
        self.class_names = []
        self.load_data(srcpath)

    def load_data(self, srcpath):
        pass

    def convert(self, dstpath, dataset_name, split, num_shards):
        pass

    def print_progress(self, count, total):
        # Percentage completion.
        pct_complete = float(count) / total

        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    def wrap_int64(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def wrap_bytes(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class KittiEigenSplit(TfrecordMaker):
    def __init__(self, srcpath, test_src_dir):
        self.test_src_dir = test_src_dir
        super().__init__(srcpath)

    # TODO
    # step1. 여기까지 잘 되는지 확인 - done
    # step2. 상대포즈 계산하고 확인 - done
    # step3. 폴더 단위가 아니라 파일단위까지 내려가는 리스트 만들기
    # step4. convert() train dataset 만들기
    # step5. test_files_eigen.txt 로 테스트 파일 만들기
    def load_data(self, srcpath):
        test_dirs = self.read_test_scenes()
        train_set, test_set = self.split_dataset(srcpath, test_dirs)
        self.data = {"train": train_set, "test": test_set}

    def read_test_scenes(self):
        test_scenes_txt = os.path.join(self.test_src_dir, "test_scenes_eigen.txt")
        with open(test_scenes_txt, 'r') as f:
            test_scenes = f.readlines()
            test_scenes = [scene.strip('\n') for scene in test_scenes]
        return test_scenes

    # split data scene dirs
    def split_dataset(self, dataset_root, test_scenes):
        train_set = []
        test_set_ = []

        date_dirs = os.listdir(dataset_root)
        date_dirs = [scdate for scdate in date_dirs if '2011_' in scdate]

        for scdate in date_dirs:
            scene_dirs = os.listdir(os.path.join(dataset_root, scdate))
            # scene dir must start with 'date'
            scene_dirs = [os.path.join(dataset_root, scdate, scene) for scene in scene_dirs if scdate in scene]
            # calibration file is under dateset_root/'date'
            intrin_left_, intrin_right, T_l2r \
                = self.read_camera_params(os.path.join(dataset_root, scdate))
            train_dirs, test_dirs = self.split_dirs(scene_dirs, test_scenes)
            # print("train\n", train_dirs, '\n', intrin_right, '\n', intrin_left_, '\n', T_l2r)

            for tdir in train_dirs:
                train_set.append({"scene_path": tdir, "intrin_left": intrin_left_,
                                  "intrin_right": intrin_right, "T_l2r": T_l2r})
            for tdir in test_dirs:
                test_set_.append({"scene_path": tdir, "intrin_left": intrin_left_,
                                  "intrin_right": intrin_right, "T_l2r": T_l2r})

        return train_set, test_set_

    def read_camera_params(self, path):
        filepath = os.path.join(path, "calib_cam_to_cam.txt")
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # TODO: 어느쪽이 오른쪽인지 확인 안했음, transformation의 방향도 확실치 않음
        intrin_left_, T_left_ = self.get_camera_param(data, '02')
        intrin_right, T_right = self.get_camera_param(data, '03')
        T_left_inv = np.linalg.inv(T_left_)
        # numpy matrix multiplication must use 'matmul' instead of * operator
        T_l2r = np.matmul(T_right, T_left_inv)
        return intrin_left_, intrin_right, T_l2r

    @staticmethod
    def get_camera_param(data, cam_id):
        intrin = np.reshape(data['P_rect_' + cam_id], (3, 4))
        intrin = intrin[:3, :3]
        R = np.reshape(data['R_' + cam_id], (3, 3))
        t = np.reshape(data['T_' + cam_id], (3, 1))
        T = np.concatenate((R, t), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
        # print('transform', cam_id, '\n', R, '\n', t, '\n', T)
        return intrin, T



    @staticmethod
    def split_dirs(scene_dirs, test_scenes):
        train_dirs = []
        test_dirs = []
        for scene_dir in scene_dirs:
            temp = [scene_name for scene_name in test_scenes if scene_name in scene_dir]
            if temp:
                test_dirs.append(scene_dir)
            else:
                train_dirs.append(scene_dir)
        return train_dirs, test_dirs

    # TODO: 본격 tfrecord 만들기
    def convert(self, dstpath, dataset_name, split, num_shards):
        data = self.data[split]
        for datum in data:
            pass






    # TODO: 이 아래는 참고용으로 남겨놓음
    def convert_old(self, dstpath, dataset_name, split, num_shards):
        images = self.data[split]["image"]
        targets = self.data[split]["target"]
        labels = self.data[split]["label"]

        assert(images.shape[2]==labels.shape[0])
        num_images = labels.shape[0]
        num_images_per_shard = int(math.ceil(num_images / num_shards))

        for si in range(num_shards):
            outfile = "{}/{}_{}_{:04d}.tfrecord".format(dstpath, dataset_name, split, si)
            print("\bConverting: " + outfile)

            with tf.python_io.TFRecordWriter(outfile) as writer:
                for mi in range(num_images_per_shard):
                    i = si*num_images_per_shard + mi
                    if i >= num_images:
                        break

                    # print the percentage-progress.
                    self.print_progress(count=i, total=num_images - 1)

                    serialized = self.make_serialized_example(images[:, :, i], targets[i], labels[i, :])
                    writer.write(serialized)

    def make_serialized_example(self, image, target, label):
        # convert the image to raw bytes.
        img_bytes = image.tostring()
        label = label.astype(np.int64)
        label_bytes = label.tostring()

        # create a dict with the data we want to save in the TFRecords file.
        data = {'image': self.wrap_bytes(img_bytes),
                'target': self.wrap_int64(target),
                'label': self.wrap_bytes(label_bytes)
                }

        # wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)

        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)

        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

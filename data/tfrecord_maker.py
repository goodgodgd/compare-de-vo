import os
import sys
from data.tfrd_feeder import *


class TfrecordsMaker:
    def __init__(self, _opt):
        self.opt = _opt

    def _load_split(self, split):
        raise NotImplementedError()
    
    def make(self, split):
        print("\n==================================================")
        opt = self.opt
        data_feeders = self._load_split(split)

        num_images = len(data_feeders[0])
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
        for feeder in feeders:
            example.update(feeder.get_next())
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
        dataset_dir = self.opt.dataset_dir
        list_file = "{}/{}.txt".format(dataset_dir, split)
        print("frame list file for {}:".format(split), list_file)
        image_list, gt_list, cam_list = self._read_filelist(dataset_dir, list_file)
        N = len(image_list)
        assert N == len(gt_list)

        gt_feeder = None
        # pose written in readable text
        if self.opt.dataset_name == "kitti_odom":
            gt_feeder = TextFeeder("gt", gt_list, dtype=np.float32)
        elif self.opt.dataset_name == "kitti_raw_eigen":
            # no depth available for train data
            if split == "train":
                gt_feeder = ConstFeeder("gt", np.array([0], dtype=np.float32), N)
            # depth is available for test data
            elif split == "test":
                gt_feeder = NpyFeeder("gt", gt_list, dtype=np.float32)

        data_feeders = [
            ImageFeeder("image", image_list, dtype=np.uint8),
            gt_feeder,
            TextFeeder("intrinsic", cam_list, dtype=np.float32, shape_out=False),
        ]
        return data_feeders

    @staticmethod
    def _read_filelist(dataset_root, list_file):
        image_files = []
        gt_files = []
        cam_files = []

        with open(list_file, 'r') as f:
            for line in f:
                paths = line.split(" ")
                seq_dir = paths[0]
                frame_id = paths[1][:-1]

                imgfile = os.path.join(dataset_root, seq_dir, frame_id+".jpg")
                image_files.append(imgfile)
                gtfile = os.path.join(dataset_root, seq_dir, "gt", frame_id+"_gt.txt")
                gt_files.append(gtfile)
                camfile = os.path.join(dataset_root, seq_dir, "intrinsics.txt")
                cam_files.append(camfile)
        return image_files, gt_files, cam_files

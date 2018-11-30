import os
import numpy as np
import tensorflow as tf
from models.geonet.geonet_feeder import dataset_feeder
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

tf.logging.set_verbosity(tf.logging.INFO)


# tf.Estimator based model operator, It is not used for test for now
class ModelOperator:
    def __init__(self, opt, _model):
        self.opt = opt
        self.model = _model
        self.estimator = self._create_estimator(opt.checkpoint_dir)

    def _create_estimator(self, ckpt_path):
        def cnn_model_fn(features, mode):
            return self._cnn_model_fn(features, mode)

        # create estimator
        return tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=ckpt_path)

    def _cnn_model_fn(self, features, mode):
        raise NotImplementedError()

    def train(self, show_log: bool=False):
        def data_feeder():
            return self.data_feeder(self.opt, "train")

        # train the model
        logging_hook = self._get_logging_hook(show_log)
        self.estimator.train(input_fn=data_feeder, hooks=logging_hook)
        print("training finished")

    def evaluate(self, show_log: bool=False):
        # currently EVAL mode is not used
        pass

    @staticmethod
    def data_feeder(opt, split):
        raise NotImplementedError()

    def predict(self, input_image, show_log: bool=False):
        pass

    def save_results(self):
        raise NotImplementedError()

    def _get_logging_hook(self, show_log):
        if show_log is False:
            return None
        return self._create_logging_hook()

    @staticmethod
    def _create_logging_hook():
        raise NotImplementedError()


class GeoNetOperator(ModelOperator):
    def __init__(self, opt, _model_builder):
        super().__init__(opt, _model_builder)
        self.tgt_idx = (self.opt.seq_length - 1) // 2

        with open(os.path.join(opt.dataset_dir, 'sequences/%.2d/times.txt'
                                                % opt.pose_test_seq), 'r') as f:
            times = f.readlines()
        self.times = np.array([float(s[:-1]) for s in times])

    def _cnn_model_fn(self, features, mode):
        opt = self.opt
        self.tgt_idx = self.tgt_idx + opt.batch_size
        src_image_stack = features["sources"]
        tgt_image = features["target"]
        intrinsics_ms = features["intrinsics_ms"]

        self.model.build_model(tgt_image, src_image_stack, intrinsics_ms)

        if mode == tf.estimator.ModeKeys.PREDICT:
            if opt.mode == "test_pose":
                # pred_pose: (batch, num_sources, 6(pose))
                pred_pose = self.model.get_pose_pred()
                dump_pose_seq(pred_pose.numpy())
            if opt.mode == "test_depth":
                pred_depth_stack = self.outputs["pred_depth"]
                pred_depth = self.model.get_depth_pred()
                print("depth predction", pred_depth.shape)
                pred_depth = pred_depth.numpy()
                pred_depth_stack = pred_depth if pred_depth_stack is None \
                                   else np.concatenate((pred_depth_stack, pred_depth), axis=0)
                self.outputs["pred_depth"] = pred_depth_stack

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = self.model.get_loss()
            # define training operation
            optimizer = tf.train.AdamOptimizer(self.opt.learning_rate, 0.9)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        if mode == tf.estimator.ModeKeys.EVAL:
            # currently EVAL mode is not used
            pass

    @staticmethod
    def data_feeder(opt, split):
        return dataset_feeder(opt, split)

    def dump_pose_seq(self, pred_poses):
        max_src_offset = (self.opt.seq_length - 1) // 2
        # Insert the target pose [0, 0, 0, 0, 0, 0]
        pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=1)

        for b in range(self.opt.batch_size):
            idx = self.tgt_idx + b
            pred_pose = pred_poses[b]
            curr_times = self.times[idx - max_src_offset:idx + max_src_offset + 1]
            out_file = os.path.join(self.opt.output_dir, '%.6d.txt' % (idx - max_src_offset))
            dump_pose_seq_TUM(out_file, pred_pose, curr_times)


    @staticmethod
    def _create_logging_hook():
        raise NotImplementedError()


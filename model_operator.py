import tensorflow as tf
import settings
from model_handler.data_feeder.dataset_feeder import dataset_input_fn

tf.logging.set_verbosity(tf.logging.INFO)

# TODO: ModelOperator는 abstract로 보내

# tf.Estimator based model trainder, It is not used for test
class ModelOperator:
    def __init__(self, opt, _model, reshaper):
        self.opt = opt
        self.model = _model
        self.estimator = self._create_estimator(opt.checkpoint_dir)
        self.reshape_input = reshaper
        self.batch_size = 32

    def _create_estimator(self, ckpt_path):
        def cnn_model_fn(features, mode):
            return self._cnn_model_fn(features, mode)

        # create estimator
        return tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=ckpt_path)

    def _cnn_model_fn(self, features, mode):
        raise NotImplementedError()

    def train(self, data_path, batch_size: int=32, epochs: int=5, show_log: bool=False):
        def data_feeder():
            return self.dataset_feeder(data_path, "train", batch_size, epochs)

        self.batch_size = batch_size
        # train the model
        logging_hook = self._get_logging_hook(show_log)
        self.estimator.train(input_fn=data_feeder, hooks=logging_hook)
        print("training finished")

    def evaluate(self, data_path, batch_size: int=32, show_log: bool=False):
        # currently EVAL mode is not used
        pass

    def dataset_feeder(self, dirpath, split, batch_size, epochs):
        raise NotImplementedError()

    def predict(self, input_image, show_log: bool=False):
        # currently EVAL mode is not used
        pass

    def _get_logging_hook(self, show_log):
        if show_log is False:
            return None
        return self._create_logging_hook()

    @staticmethod
    def _create_logging_hook():
        raise NotImplementedError()


class PosePredModel(ModelOperator):
    def __init__(self, opt, _model_builder):
        super().__init__(opt, _model_builder)

    def _cnn_model_fn(self, features, mode):
        """Model function for CNN."""
        src_image_stack = self.reshape_input(features["sources"])
        # TODO: reshape input 구현
        tgt_image = self.reshape_input(features["target"], (self.opt.img_width, self.opt.img_height))
        intrinsic = features["intrinsic"]

        # TODO: 모델 클래스 변형
        pred_poses = self.model.build_layers(tgt_image, src_image_stack, intrinsic, mode=mode)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # TODO: poses 파일로 저장
            pass

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

    # TODO: dataset_input_fn 구현
    def dataset_feeder(self, dirpath, split, batch_size, epochs):
        return dataset_input_fn(tfrecord_dirpath=dirpath, split=split, batch_size=batch_size,
                                train_epochs=epochs, data_type="mnist_multi")

    @staticmethod
    def _create_logging_hook():
        raise NotImplementedError()

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# tf.Estimator based model operator, It is not used for test for now
class ModelOperator:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.estimator = self._create_estimator(opt.checkpoint_dir)

    def _create_estimator(self, ckpt_path):
        def cnn_model_fn(features, mode):
            return self._cnn_model_fn(features, mode)

        # create estimator
        return tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=ckpt_path)

    def _cnn_model_fn(self, features, mode):
        raise NotImplementedError()

    def train(self, data_feeder, show_log: bool=False):
        # train the model
        logging_hook = self._get_logging_hook(show_log)
        self.estimator.train(input_fn=data_feeder, hooks=logging_hook)
        print("training finished")

    def evaluate(self, data_feeder, show_log: bool=False):
        # evaluate the model
        logging_hook = self._get_logging_hook(show_log)
        eval_results = self.estimator.evaluate(input_fn=data_feeder, hooks=logging_hook)
        return eval_results

    def predict(self, data_feeder):
        # predict classification result by the model
        pred_result = self.estimator.predict(input_fn=data_feeder)
        pred_result = list(pred_result)[0]
        return pred_result

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

    def _cnn_model_fn(self, features, mode):
        src_image_stack = features["sources"]
        tgt_image = features["target"]
        intrinsics_ms = features["intrinsics_ms"]

        self.model.build_model(tgt_image, src_image_stack, intrinsics_ms)

        prediction = None
        if self.opt.mode == "test_pose":
            prediction = self.model.get_pose_pred()
        elif self.opt.mode == "test_eigen":
            prediction = self.model.get_depth_pred()
        # format return type of estimator.predict()
        predictions = {"prediction": prediction}
        loss = self.model.get_loss()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # define training operation
            optimizer = tf.train.AdamOptimizer(self.opt.learning_rate, 0.9)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = predictions
            return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=eval_metric_ops)

    @staticmethod
    def _create_logging_hook():
        return None


import tensorflow as tf


# !!! when executing eagerly, return 'iterable' dataset
# !!! when normal session, return 'tensors' of features and labels
# guide line in https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data
def dataset_feeder(opt, split="train"):
    # 파일 패턴으로 파일 리스트 입력
    file_pattern = ["{}/*_{}_*.tfrecord".format(opt.tfrecords_dir, split)]
    filenames = tf.gfile.Glob(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames)

    def parse_example_opt(record):
        return parse_example(record, opt.img_height, opt.img_width, opt.seq_length, opt.num_scales)

    # use `Dataset.map()` to build a pair of feature dictionary and label tensor for each example.
    dataset = dataset.map(parse_example_opt)
    return dataset_process(dataset, split, opt.batch_size, opt.train_epochs)


# use `tf.parse_single_example()` to extract data from a `tf.Example` protocol buffer,
# and perform any additional per-record preprocessing.
def parse_example(record, img_height, img_width, seq_length, num_scales):
    # np.to_string()으로 ndarray 이미지를 string으로 변환한 다음 저장했기 때문에 tf.string으로 불러옴
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "intrinsic": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # perform additional preprocessing on the parsed data.
    # string to numeric type
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.reshape(image, shape=(img_height, img_width*seq_length, 3))
    tgt_image, src_image_stack = unpack_image_sequence(image, img_height, img_width, seq_length)

    intrinsic = tf.decode_raw(parsed["intrinsic"], tf.float32)
    intrinsic = tf.reshape(intrinsic, shape=(3, 3))
    intrinsics_ms = get_multi_scale_intrinsics(intrinsic, num_scales)
    return {"target": tgt_image, "sources": src_image_stack, "intrinsics_ms": intrinsics_ms}


def unpack_image_sequence(image_seq, img_height, img_width, seq_length):
    num_sources = seq_length - 1
    target_ind = int(num_sources // 2)
    # Assuming the center image is the target frame
    tgt_start_idx = int(img_width * target_ind)
    tgt_image = tf.slice(image_seq,
                         [0, tgt_start_idx, 0],
                         [-1, img_width, -1])
    # Source frames before the target frame
    src_image_1 = tf.slice(image_seq,
                           [0, 0, 0],
                           [-1, int(img_width * target_ind), -1])
    # Source frames after the target frame
    src_image_2 = tf.slice(image_seq,
                           [0, int(tgt_start_idx + img_width), 0],
                           [-1, int(img_width * target_ind), -1])
    src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
    # Stack source frames along the color channels (i.e. [H, W, N*3])
    src_image_stack = tf.concat([tf.slice(src_image_seq,
                                [0, i*img_width, 0],
                                [-1, img_width, -1])
                                for i in range(num_sources)], axis=2)
    src_image_stack.set_shape([img_height,
                               img_width,
                               num_sources * 3])
    tgt_image.set_shape([img_height, img_width, 3])
    return tgt_image, src_image_stack


def get_multi_scale_intrinsics(intrinsic, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        scaled_upper = tf.div(intrinsic[:2, :], 2 ** s)
        const_lower = tf.constant([[0, 0, 1]], dtype=tf.float32)
        scaled_intrin = tf.concat([scaled_upper, const_lower], axis=0)
        intrinsics_mscale.append(scaled_intrin)
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=0)
    return intrinsics_mscale


# guide line in https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data
def dataset_process(dataset, split="train", batch_size=32, train_epochs=1):
    if split.lower() is "train":
        dataset = dataset.shuffle(buffer_size=5000)
        num_epochs = train_epochs
    else:
        # If testing, only go through the data once.
        num_epochs = 1

    dataset = dataset.repeat(num_epochs)
    # drop_remainder: dataset이 끝날때 batch_size 이하로 남은 데이터는 쓰지 않고 버린다
    # -> 항상 일정한 batch_size 유지
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    if tf.executing_eagerly():
        return dataset
    else:
        iterator = dataset.make_one_shot_iterator()
        # `features` is a dictionary in which each value is a batch of values for that feature
        # `labels` is a batch of labels.
        features = iterator.get_next()
        return features

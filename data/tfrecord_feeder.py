import tensorflow as tf


# !!! when executing eagerly, return 'iterable' dataset
# !!! when normal session, return 'tensors' of features and labels
# guide line in https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data
def dataset_feeder(opt, split):
    # 파일 패턴으로 파일 리스트 입력
    file_pattern = ["{}/*_{}_*.tfrecord".format(opt.tfrecords_dir, split)]
    filenames = tf.gfile.Glob(file_pattern)
    print("========== dataset_feeder finding tfrecords:", file_pattern, filenames)
    dataset = tf.data.TFRecordDataset(filenames)

    def parse_example_opt(record):
        return parse_example(record, opt.num_scales, opt.seq_length, opt.img_height, opt.img_width)
    # use `Dataset.map()` to build a pair of feature dictionary and label tensor for each example.
    dataset = dataset.map(parse_example_opt)
    return dataset_process(dataset, split, opt.batch_size, opt.train_epochs)


# use `tf.parse_single_example()` to extract data from a `tf.Example` protocol buffer,
# and perform any additional per-record preprocessing.
def parse_example(record, num_scales, seq_length, image_height, image_width):
    # np.to_string()으로 ndarray 이미지를 string으로 변환한 다음 저장했기 때문에 tf.string으로 불러옴
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "image_shape": tf.FixedLenFeature((), tf.string, default_value=""),
        "gt": tf.FixedLenFeature((), tf.string, default_value=""),
        "gt_shape": tf.FixedLenFeature((), tf.string, default_value=""),
        "intrinsic": tf.FixedLenFeature((), tf.string, default_value=""),
        "frame": tf.FixedLenFeature((), tf.string, default_value=""),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # parse metadata
    im_shape = tf.decode_raw(parsed["image_shape"], tf.int32)
    gt_shape = tf.decode_raw(parsed["gt_shape"], tf.int32)

    # parse main data
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.reshape(image, shape=im_shape)
    gtruth = tf.decode_raw(parsed["gt"], tf.float32)
    gtruth = tf.reshape(gtruth, shape=gt_shape)
    intrinsic = tf.decode_raw(parsed["intrinsic"], tf.float32)
    intrinsic = tf.reshape(intrinsic, shape=(3, 3))
    frame_int8 = tf.decode_raw(parsed["frame"], tf.int8)

    # perform additional preprocessing on the parsed data.
    tgt_image, src_image_stack = unpack_image_sequence(image, image_height,
                                                       image_width, seq_length)
    intrinsics_ms = get_multi_scale_intrinsics(intrinsic, num_scales)

    return {"target": tgt_image, "sources": src_image_stack,
            "gt": gtruth, "intrinsics_ms": intrinsics_ms, "frame_int8": frame_int8}


def unpack_image_sequence(image_seq, img_height, img_width, seq_length):
    if seq_length == 1:
        return image_seq, image_seq

    tgt_ind = (seq_length - 1) // 2
    # assuming the center image is the target frame
    tgt_image = tf.slice(image_seq,
                         [0, tgt_ind * img_width, 0],
                         [-1, img_width, -1])
    # stack all other images as src_image_stack
    src_image_stack = []
    for src_ind in range(seq_length):
        if src_ind != tgt_ind:
            src_image = tf.slice(image_seq,
                                 [0, src_ind * img_width, 0],
                                 [-1, img_width, -1])
            src_image_stack.append(src_image)
    src_image_stack = tf.concat(src_image_stack, axis=2)

    src_image_stack.set_shape([img_height,
                               img_width,
                               (seq_length - 1) * 3])
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
def dataset_process(dataset, split, batch_size, train_epochs):
    if split.lower() == "train":
        dataset = dataset.shuffle(buffer_size=5000)
        num_epochs = train_epochs
    else:
        # If testing, only go through the data once.
        num_epochs = 1
    print("========== dataset_feeder num_epochs:", num_epochs)
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

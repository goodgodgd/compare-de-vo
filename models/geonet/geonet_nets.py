# The network design is based on Tinghui Zhou & Clement Godard's works:
# https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from models.geonet.inception_v4 import inception_v4_base

# Range of disparity/inverse depth values
DISP_SCALING_RESNET50 = 5
DISP_SCALING_VGG = 10
FLOW_SCALING = 0.1


def disp_net(opt, dispnet_inputs):
    is_training = opt.mode == 'train_rigid'
    if opt.feat_net == 'vgg':
        return build_vgg(dispnet_inputs, get_disp_vgg, is_training, 'depth_net')
    elif opt.feat_net == 'resnet50':
        return build_resnet50(dispnet_inputs, get_disp_resnet50, is_training, 'depth_net')
    elif opt.feat_net == 'inceptionv4':
        return build_inceptionv4(dispnet_inputs, get_disp_resnet50, is_training, 'depth_net')


def flow_net(opt, flownet_inputs):
    is_training = opt.mode == 'train_flow'
    return build_resnet50(flownet_inputs, get_flow, is_training, 'flow_net')


def pose_net(opt, posenet_inputs):
    is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('pose_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1  = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2  = slim.conv2d(conv1, 32,  5, 2)
            conv3  = slim.conv2d(conv2, 64,  3, 2)
            conv4  = slim.conv2d(conv3, 128, 3, 2)
            conv5  = slim.conv2d(conv4, 256, 3, 2)
            conv6  = slim.conv2d(conv5, 256, 3, 2)
            conv7  = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 6*opt.num_source, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 6])
            return pose_final


def pose_net_inceptionv4(opt, posenet_inputs):
    with tf.variable_scope('pose_net') as sc:
        net, endpoints = inception_v4_base(posenet_inputs)
        print("==========build posenet by inception v4")
        pose_pred = slim.conv2d(net, 6 * opt.num_source, 1, 1,
                                normalizer_fn=None, activation_fn=None)
        pose_avg = tf.reduce_mean(pose_pred, [1, 2])
        pose_final = 0.01 * tf.reshape(pose_avg, [-1, opt.num_source, 6])
        return pose_final


def build_vgg(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    with tf.variable_scope(var_scope) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            # ENCODING
            conv1  = slim.conv2d(inputs, 32,  7, 2)
            conv1b = slim.conv2d(conv1,  32,  7, 1)
            conv2  = slim.conv2d(conv1b, 64,  5, 2)
            conv2b = slim.conv2d(conv2,  64,  5, 1)
            conv3  = slim.conv2d(conv2b, 128, 3, 2)
            conv3b = slim.conv2d(conv3,  128, 3, 1)
            conv4  = slim.conv2d(conv3b, 256, 3, 2)
            conv4b = slim.conv2d(conv4,  256, 3, 1)
            conv5  = slim.conv2d(conv4b, 512, 3, 2)
            conv5b = slim.conv2d(conv5,  512, 3, 1)
            conv6  = slim.conv2d(conv5b, 512, 3, 2)
            conv6b = slim.conv2d(conv6,  512, 3, 1)
            # revise from GeoNet: to upconv only 6 levels: stride=1
            conv7  = slim.conv2d(conv6b, 512, 3, 1)
            conv7b = slim.conv2d(conv7,  512, 3, 1)

            features = [conv1b, conv2b, conv3b, conv4b, conv5b, conv7b]
            return make_scaled_depth_preds(features, get_pred)


def build_resnet50(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = conv(inputs, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D
            features = [conv1, pool1, conv2, conv3, conv4, conv5]
            for i, feat in enumerate(features):
                print("``````````resnet50 features:", i, feat.get_shape())

            return make_scaled_depth_preds(features, get_pred)


def build_inceptionv4(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope) as sc:
        net, endpoints = inception_v4_base(inputs)
        features = list()
        features.append(endpoints["Conv2d_2b_3x3"])
        features.append(endpoints["Mixed_3a"])
        features.append(endpoints["Mixed_4a"])
        features.append(endpoints["Mixed_5e"])
        features.append(endpoints["Mixed_6h"])
        features.append(endpoints["Mixed_7d"])
        for i, feat in enumerate(features):
            print("``````````inception v4 features:", i, feat.get_shape())

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            return make_scaled_depth_preds(features, get_pred)


def make_scaled_depth_preds(features, get_pred):
    # DECODING
    upconv6 = upconv(features[5], 512, 3, 2)  # H/32
    upconv6 = resize_like(upconv6, features[4])
    concat6 = tf.concat([upconv6, features[4]], 3)
    iconv6 = conv(concat6, 512, 3, 1)

    upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
    upconv5 = resize_like(upconv5, features[3])
    concat5 = tf.concat([upconv5, features[3]], 3)
    iconv5 = conv(concat5, 256, 3, 1)

    upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
    upconv4 = resize_like(upconv4, features[2])
    concat4 = tf.concat([upconv4, features[2]], 3)
    iconv4 = conv(concat4, 128, 3, 1)
    pred4 = get_pred(iconv4)
    upred4 = upsample_nn(pred4, 2)

    upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
    concat3 = tf.concat([upconv3, features[1], upred4], 3)
    iconv3 = conv(concat3, 64, 3, 1)
    pred3 = get_pred(iconv3)
    upred3 = upsample_nn(pred3, 2)

    upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
    concat2 = tf.concat([upconv2, features[0], upred3], 3)
    iconv2 = conv(concat2, 32, 3, 1)
    pred2 = get_pred(iconv2)
    upred2 = upsample_nn(pred2, 2)

    upconv1 = upconv(iconv2, 16, 3, 2)  # H
    concat1 = tf.concat([upconv1, upred2], 3)
    iconv1 = conv(concat1, 16, 3, 1)
    pred1 = get_pred(iconv1)

    return [pred1, pred2, pred3, pred4]


def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)


def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)


def get_disp_vgg(x):
    disp = DISP_SCALING_VGG * slim.conv2d(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp


def get_disp_resnet50(x):
    disp = DISP_SCALING_RESNET50 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp


def get_flow(x):
    # Output flow value is normalized by image height/width
    flow = FLOW_SCALING * slim.conv2d(x, 2, 3, 1, activation_fn=None, normalizer_fn=None)
    return flow


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv


def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.relu(conv3 + shortcut)


def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

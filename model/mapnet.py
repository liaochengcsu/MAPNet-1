# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.layers import UpSampling2D


def conv2d(input,filters,kernel_size=3,stride=1,padding='SAME'):
    return tf.layers.conv2d(input,filters=filters,kernel_size=kernel_size,
                            padding=padding,strides=stride,use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer())


def bn(input,is_training=True):
    return tf.layers.batch_normalization(input,momentum=0.1,epsilon=1e-5,training=is_training)


def bottleneck(x, size,is_training,downsampe=False):
    residual = x
    out = bn(x, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 1, padding='VALID')
    out = bn(out, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    out = bn(out, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size * 4, 1, padding='VALID')

    if downsampe:
        residual = bn(x, is_training)
        residual = tf.nn.relu(residual)
        residual = conv2d(residual, size * 4, 1, padding='VALID')
    out = tf.add(out,residual)

    return out


def resblock(x, size,is_training):
    residual = x
    out = bn(x, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    out = bn(out, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)

    out = tf.add(out, residual)
    return out


def stage0(x,is_training):
    x = bottleneck(x, 64,is_training,downsampe=True)
    x = bottleneck(x, 64,is_training)
    x = bottleneck(x, 64,is_training)
    x = bottleneck(x, 64,is_training)

    return x


def transition_layer(x, in_channels, out_channels,is_training):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []

    for i in range(num_out):
        if i < num_in:

            residual = bn(x[i], is_training)
            residual = tf.nn.relu(residual)
            residual = conv2d(residual, out_channels[i], 3)

            out.append(residual)
        else:

            residual = bn(x[-1], is_training)
            residual = tf.nn.relu(residual)
            residual = conv2d(residual, out_channels[i], 3, stride=2)

            out.append(residual)

    return out


def convb(x, block_num, channels,is_training):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = resblock(residual, channels[i],is_training)
        out.append(residual)
    return out


def featfuse(x, channels, is_training, multi_scale_output=True):
    out = []

    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]

        for j in range(len(channels)):
            if j > i:
                if multi_scale_output == False:
                    y = bn(x[j], is_training)
                    y = tf.nn.relu(y)
                    y = conv2d(y, channels[j], 1, padding='VALID')

                    out.append(UpSampling2D(size=2 ** (j - i))(y))
                else:
                    y = bn(x[j], is_training)
                    y = tf.nn.relu(y)
                    y = conv2d(y, channels[i], 1, padding='VALID')

                    y = UpSampling2D(size=2 ** (j - i))(y)
                    residual = tf.add(residual, y)

            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = bn(y, is_training)
                        y = tf.nn.relu(y)
                        # y = conv2d(y, channels[i], 3, stride=2)
                        y = conv2d(y, channels[i], 1)
                        y = tf.layers.max_pooling2d(y, 2, 2)

                    else:
                        y = bn(y, is_training)
                        y = tf.nn.relu(y)
                        # y = conv2d(y, channels[j], 3, stride=2)
                        y = conv2d(y, channels[j], 1)
                        y = tf.layers.max_pooling2d(y, 2, 2)

                residual = tf.add(residual, y)
        # residual = tf.nn.relu(residual)
        out.append(residual)

    return out


def convblock(x, channels,is_training,multi_scale_output=True):
    residual = convb(x, 4, channels,is_training)
    out = featfuse(residual, channels,is_training,
                      multi_scale_output=multi_scale_output)
    return out


def stage(x, num_modules, channels, is_training,multi_scale_output=True):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = convblock(out, channels,is_training, multi_scale_output=False)
        else:
            out = convblock(out, channels,is_training)

    return out


def channel_squeeze(input,filters,ratio,name=" "):
    with tf.name_scope(name):
        squeeze=tf.reduce_mean(input,axis=[1,2])
        with tf.name_scope(name+"fc1"):
            fc1=tf.layers.dense(squeeze,use_bias=True,units=filters)
            fc1=tf.nn.relu(fc1)
        with tf.name_scope(name+"fc2"):
            fc2=tf.layers.dense(fc1,use_bias=True,units=filters)
            fc2=tf.nn.sigmoid(fc2)
        result=tf.reshape(fc2,[-1,1,1,filters])
        return input*result


def spatial_pooling(input):
    h,w=input.shape[1],input.shape[2]
    p1=tf.image.resize_bilinear(tf.layers.max_pooling2d(input,2,2),(h,w))
    p2 = tf.image.resize_bilinear(tf.layers.max_pooling2d(input, 3, 3), (h, w))
    p3=tf.image.resize_bilinear(tf.layers.max_pooling2d(input,5,5),(h,w))
    p4 = tf.image.resize_bilinear(tf.layers.max_pooling2d(input, 6, 6), (h, w))
    p=tf.concat([p1,p2,p3,p4,input],axis=-1)
    return p


def mapnet(input,is_training=True):
    channels_2 = [64, 128]
    channels_3 = [64, 128, 256]
    num_modules_2 = 2
    num_modules_3 = 3

    # input=tf.image.per_image_standardization(input)
    conv_1 = conv2d(input, 64)
    conv_1 = bn(conv_1, is_training)
    conv_1 = tf.nn.relu(conv_1)
    conv_2 = conv2d(conv_1, 64)
    conv_2=tf.layers.max_pooling2d(conv_2, 2, 2)
    conv_2 = bn(conv_2, is_training)
    conv_2 = tf.nn.relu(conv_2)
    conv_3 = conv2d(conv_2, 64)
    conv_3 = bn(conv_3, is_training)
    conv_3 = tf.nn.relu(conv_3)
    conv_4 = conv2d(conv_3, 64)
    conv_4 = bn(conv_4, is_training)
    conv_4 = tf.nn.relu(conv_4)
    x = tf.layers.max_pooling2d(conv_4, 2, 2)

    la1 = stage0(x,is_training)
    tr1 = transition_layer([la1], [256], channels_2,is_training)

    st2 = stage(tr1, num_modules_2, channels_2,is_training)
    tr2 = transition_layer(st2, channels_2, channels_3,is_training)

    st3 = stage(tr2, num_modules_3, channels_3,is_training,multi_scale_output=False)

    st=tf.concat(st3,axis=-1)


    print(st3)

    seqeese=channel_squeeze(st, st.shape[-1], 1, name="se_leyaer")

    st0=tf.concat([st3[0],st3[1]],axis=-1)
    st0=spatial_pooling(st0)
    new_feature = tf.concat([st0, seqeese], axis=-1)


    new_feature = bn(new_feature, is_training)
    new_feature = tf.nn.relu(new_feature)
    result=conv2d(new_feature, 128, 1, padding='SAME')
    up1=tf.image.resize_bilinear(result,size=(st3[0].shape[1]*2,st3[0].shape[2]*2))
    print(up1)
    up1 = bn(up1, is_training)
    up1 = tf.nn.relu(up1)
    up1 = conv2d(up1, 64, 3)

    up2 = tf.image.resize_bilinear(up1, size=(up1.shape[1]*2, up1.shape[2]*2))
    print(up2)
    up2 = bn(up2, is_training)
    up2 = tf.nn.relu(up2)
    up2 = conv2d(up2, 32, 3)

    up2 = bn(up2, is_training)
    up2 = tf.nn.relu(up2)
    final = conv2d(up2, 1, 1, padding='VALID')

    return final
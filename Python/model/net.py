import tensorflow as tf

SCALE = 1.0

def conv_net_V7a_3d(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        is_train = features["is_train"]
        data = tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=144,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=192,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=192,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 4 * 192])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=6*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        print(fc1.shape)
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        fc2 = tf.layers.dense(inputs=dropout_fc1, units=6*32, use_bias=True, activation=None, name='layer5_fc')
        print(fc2.shape)
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')
        print(fc3.shape)

        ret = tf.identity(fc3, name='model')

    return ret


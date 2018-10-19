import tensorflow as tf

def coor_conv2d(x,filters,size=(3,3),padding='SAME',trainable=True):
    shape=x.get_shape().as_list()
    xx_ones=tf.ones([cfg.batch_size,shape[1]],dtype=tf.int32)
    xx_ones=tf.expand_dims(xx_ones,-1)
    xx_range=tf.tile(tf.expand_dims(tf.range(shape[1]),0),[cfg.batch_size,1])
    xx_range=tf.expand_dims(xx_range,1)
    xx_channel=tf.matmul(xx_ones,xx_range)
    xx_channel=tf.expand_dims(xx_channel,-1)

    yy_ones = tf.ones([cfg.batch_size, shape[2]], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, -1)
    yy_range = tf.tile(tf.expand_dims(tf.range(shape[2]), 0), [cfg.batch_size, 1])
    yy_range = tf.expand_dims(yy_range, 1)
    yy_channel = tf.matmul(yy_ones, yy_range)
    yy_channel = tf.expand_dims(yy_channel, -1)

    xx_channel=tf.cast(xx_channel,tf.float32)/(shape[1]-1)
    yy_channel = tf.cast(yy_channel, tf.float32) / (shape[2] - 1)
    xx_channel=xx_channel*2-1
    yy_channel=yy_channel*2-1

    ret=tf.concat([x,xx_channel,yy_channel],axis=-1)
    return tf.layers.conv2d(ret,filters=filters,kernel_size=(size,size),padding=padding,activation=tf.nn.relu,trainable=trainable)

def residual_block(x,n_out, is_training=True, scope='res_block'):
    with tf.variable_scope(scope):
        if x.get_shape().as_list()[-1]==n_out:
            shortcut=tf.identity(x,name='shortcut')
        else:
            shortcut=tf.layers.conv2d(x, n_out, [1,1], padding='VALID', use_bias=False)
        y=tf.layers.conv2d(x,n_out,[3,3],padding='SAME',use_bias=False)
        y = tf.layers.batch_normalization(y,training=is_training)
        y = tf.nn.relu(y,name='relu_1')
        y = tf.layers.conv2d(y, n_out, [3, 3], padding='SAME', use_bias=False)
        y = tf.layers.batch_normalization(y,training=is_training)
        y = y + shortcut
        y = tf.nn.relu(y, name='relu_2')
    return y

def batch_norm(x, train_phase, decay=0.9, eps=1e-5):
    shape = x.get_shape().as_list()

    assert len(shape) in [2, 4]

    n_out = shape[-1]
    beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0))
    gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.constant_initializer(1))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(tf.cast(train_phase,tf.bool), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

def conv2d(name, x, maps_in, maps_out, train_phase, size=3, stride=1, act=tf.nn.relu,padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d", shape=[size, size, maps_in, maps_out], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        c = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        #b = tf.get_variable(name="conv2d_b", shape=[maps_out], initializer=tf.constant_initializer(0))
        bn = batch_norm(c, train_phase)
        #bn = b + c

    if act is not None:
        return act(bn, name=name)
    else:
        return bn

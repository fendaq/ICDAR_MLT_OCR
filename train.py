import tensorflow as tf
import numpy as np
from config import args

img= tf.placeholder(tf.float32, shape=[None,args.img_height,args.img_width,args.img_ch])
label=tf.placeholder(tf.float32, shape=[None,args.max_len_word])

train_dataset = tf.data.Dataset.from_tensor_slices((img,label)).batch(args.batch_size).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((img,label)).batch(args.batch_size).repeat()

train_data = (np.random.sample((100,)), np.random.sample((100,1)))
val_data = (np.random.sample((20,2)), np.random.sample((20,1)))

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
x, y = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(val_dataset)

# make a simple model
net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(train_init_op, feed_dict = {x : train_data[0], y: train_data[1]})
    print('Training...')
    for i in range(args.epoch):
        tot_loss = 0
        n_batches=10
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    # initialise iterator with test data
    sess.run(test_init_op, feed_dict = {x : val_data[0], y: val_data[1]})
    print('Test Loss: {:4f}'.format(sess.run(loss)))
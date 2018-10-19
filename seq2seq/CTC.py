import tensorflow as tf

def ctc_layer(inputs,labels,seq_len,num_class=100):
    ''' get ctc loss and output

    :param inputs: the output of cnn or LSTM,shape should be[batch,time_step,num_hidden]
    :param num_class: the number of class
    :return: ctc_loss,predict
    '''
    logits=tf.layers.dense(inputs,units=num_class,activation=None)
    logits=tf.transpose(logits,(1,0,2))
    ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=seq_len,time_major=True)
    loss = tf.reduce_mean(ctc_loss)
    decoded, log_prob=tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=True,beam_width=10)
    dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
    return loss,dense_decoded
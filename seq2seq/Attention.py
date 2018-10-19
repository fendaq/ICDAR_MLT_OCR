import tensorflow as tf

class Attention_LSTM(object):
    def __init__(self,num_units,max_num_decode,vocab_size,
                 embed_dim,start_token=134,end_token=135,is_train=True,batch_size=64):
        self.num_units=num_units
        self.max_decode_length=max_num_decode
        self.end_token=1
        self.start_token=start_token
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.is_train=is_train
        self.last_output=None
        self.last_state=None
        self.last_word=None
        self.batch_size=batch_size




    def attend(self,output,reshaped_contexts,num_ctx,last_alpha,idx,reuse=None):
        """ Attention Mechanism. """
        with tf.variable_scope('attend_t%s'%idx,reuse=reuse):
            output=tf.expand_dims(output,axis=1)
            #last_alpha=tf.expand_dims(last_alpha,axis=2)
            #print(last_alpha)
            tile_h=tf.tile(output,[1,num_ctx,1])
            concat_context=tf.concat((reshaped_contexts,tile_h),-1)
            logits1=tf.layers.dense(concat_context,units=128,activation=tf.nn.tanh)
            logits1=tf.layers.dense(logits1,units=1,use_bias=False,activation=None)
            logits1 = tf.reshape(logits1, [-1, num_ctx])
            alpha=tf.nn.softmax(logits1)

        return alpha


    def output(self,contexts,initial_state=None,masks=None,sentences=None):
        '''

        :param contexts: (batch_size,4,time,feature)
        :param initial_state:
        :param masks:
        :param sentences:
        :return:
        '''
        alphas=[]
        predictions=[]
        cross_entropies=[]

        self.last_state=initial_state
        dim_ctx = contexts[0].get_shape().as_list()[-1]
        num_ctx = contexts[0].get_shape().as_list()[2]
        batch_size=self.batch_size
        alpha=tf.zeros(shape=[batch_size,num_ctx],dtype=tf.float32)
        reshaped_contexts = contexts#tf.reshape(contexts, [-1, dim_ctx])
        self.last_word = tf.zeros([batch_size] , tf.int32)+ self.start_token

        self.last_output=tf.zeros(shape=(batch_size,self.num_units))
        if masks is None and self.is_train:
            #masks=tf.cast(tf.sequence_mask(batch_size * [max_len-1], max_len),tf.float32)
            masks=tf.where(tf.equal(sentences,133),x=sentences-sentences,y=sentences-sentences+1)#ignord <PAD>
            masks=tf.cast(masks,tf.float32)
       # with tf.variable_scope("word_embedding"):
        #    embedding_matrix = tf.get_variable(name='weights',shape=[self.vocab_size, self.embed_dim],trainable=self.is_train)
        #lstm=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        lstm=tf.contrib.rnn.LSTMCell(self.num_units,cell_clip=10.,initializer=tf.orthogonal_initializer)
        with tf.variable_scope('reuse_scope') as reuse_scope:
            for idx in range(self.max_decode_length):

                with tf.variable_scope("attend") as att_scope:
                    list_contex=[]
                    num=0
                    for single_contex in reshaped_contexts:
                        single_contex=tf.squeeze(single_contex,axis=1)
                        alpha = self.attend(output=self.last_output,reshaped_contexts=single_contex,num_ctx=num_ctx,last_alpha=alpha,idx=0,reuse=num!=0 or idx!=0)
                        context = tf.reduce_sum(single_contex * tf.expand_dims(alpha, 2),axis=1)
                        list_contex.append(context)
                        #att_scope.reuse_variables()
                        #reshaped_contexts=tf.reshape(reshaped_contexts,[-1,num_ctx,dim_ctx])#*tf.expand_dims(1-alpha, 2)
                        alphas.append(alpha)
                        num+=1
                    context=tf.concat(list_contex,axis=-1)
                    #print(context)
                    # Embed the last word
                with tf.variable_scope("word_embedding"):
                    #word_embed = tf.nn.embedding_lookup(embedding_matrix,self.last_word)
                    word_embed=tf.one_hot(self.last_word,depth=self.vocab_size)
                with tf.variable_scope("lstm"):
                    current_input = tf.concat([context, word_embed], 1)
                    output, state = lstm(current_input, self.last_state)
                    memory, _ = state
                with tf.variable_scope("decode"):
                    expanded_output = tf.concat([output,context,word_embed],axis=1)
                    logits = self.decode(expanded_output)
                    probs = tf.nn.softmax(logits)
                    prediction = tf.argmax(probs, 1)
                    self.last_word = prediction
                    #if prediction==self.end_token:
                     #   break
                    predictions.append(prediction)
                if self.is_train:
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sentences[:, idx],logits=logits)
                    masked_cross_entropy = cross_entropy * masks[:, idx]
                    cross_entropies.append(masked_cross_entropy)

                self.last_output = output
                self.last_state = state

                reuse_scope.reuse_variables()
        predictions=tf.stack(predictions,axis=1)
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_mean(cross_entropies)#tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis=1)
            alphas = tf.reshape(alphas, [batch_size, num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis=2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = 0
            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + attention_loss #+ reg_loss

            return total_loss,predictions,sentences
        else:
            return 0,predictions,0




    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        logits = tf.layers.dense(expanded_output,units=self.vocab_size,activation=None,name='fc')

        return logits

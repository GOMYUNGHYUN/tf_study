import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    def __init__(self, seqLength,maxWordLen, numLabel, numTfFeature, numEMBFeature,Train_X, Train_Y, Test_X, Test_Y, Val_X, Val_Y,gloveEmbedMat,Train_tf_idf,Test_tf_idf,Val_tf_idf):
        self.numLabels=numLabel
        self.maxLen=maxWordLen
        self.seqLen=seqLength
        self.numTfFeatures=numTfFeature
        self.numFeatures=numEMBFeature
        self.train_data_x=Train_X
        self.train_data_y=Train_Y
        self.test_data_x=Test_X
        self.test_data_y=Test_Y
        self.val_data_x=Val_X
        self.val_data_y=Val_Y
        
        self.embeddingMat=gloveEmbedMat
        
        self.train_tfidf_x=Train_tf_idf
        self.test_tfidf_x=Test_tf_idf
        self.val_tfidf_x=Val_tf_idf
        
        ##
        self.hidden_size=100
        self.fc1_hidden_size=100
        self.fc2_hidden_size=100
        self.learning_rate=0.01
        self.drop_out=0.7
        
        self.output_path='data/temp2/LOG'
        self.model_output='data/temp2/MODEL'
    
    def addPlaceholder(self):
        # word_idx vect ~ [ None, sentence Number * length, numFeatures ]
        self.place_x = tf.placeholder(tf.int32, [None, self.maxLen])
        
        self.seq_len= tf.placeholder(tf.int32, shape=[None])
        
        self.tf_idf_x = tf.placeholder(tf.float32, [None,self.numTfFeatures])
        
        self.place_y = tf.placeholder(tf.int32, [None, self.numLabels])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],name="lr")
        self.dropout_ = tf.placeholder(dtype=tf.float32, shape=[],name="dropout")
    
    def addVarWordEmbedding(self):
        with tf.variable_scope("word_embedd"):
            self.variable_word_embeddings = tf.Variable(self.embeddingMat, name="var_word_embeddings", dtype=tf.float32, trainable=True)
            self.word_embeddings = tf.nn.embedding_lookup(self.variable_word_embeddings, self.place_x, name="word_embeddings_mat")
        
    # use only embedding vect
    def LogitOP(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings, self.seq_len,dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            
            output_lstm = tf.nn.dropout(output,self.dropout_) # [batch , maxLen, hidden_size*2]

            # # add channel dimension for cnn input
            # output_lstm = tf.expand_dims(output_lstm, -1) # [batch , maxLen, hidden_size*2, 1]
            # pooled = tf.nn.avg_pool(
            #                 output_lstm,
            #                 ksize=[1,self.maxLen-1, 1, 1],
            #                 strides=[1, self.maxLen, 1, 1],
            #                 padding='VALID',
            #                 name="pool") # (batch, 1, hidden_size*2, 1)
            # lstm_layer_output = tf.reshape(pooled, [-1,  2 * self.hidden_size]) # (batch, hidden_size*2)

            w = tf.Variable(tf.truncated_normal([self.maxLen * self.hidden_size*2, self.maxLen], stddev=0.1), name="w")

            b = tf.Variable(tf.truncated_normal([1, self.maxLen], stddev=0.1), name="b")
            output_lstm=tf.reshape(output_lstm,[-1,self.maxLen * self.hidden_size*2])

            lstm_layer_output =  tf.nn.relu(tf.matmul(output_lstm, w) + b) # (None, maxLen)
        
        with tf.variable_scope("lstm_fc1"):
            #tf.Variable(tf.truncated_normal([self.maxLen * self.hidden_size*2, self.maxLen], stddev=0.1), name="w")
            self.weights = tf.Variable(tf.truncated_normal([self.maxLen, self.fc1_hidden_size], stddev=0.1), name="weights")
            
            self.bias =  tf.Variable(tf.truncated_normal([1,self.fc1_hidden_size], stddev=0.1), name="bias")
            
            #self.weights = tf.Variable(tf.random_normal([2 * self.hidden_size, self.fc1_hidden_size],mean=0,stddev=(np.sqrt(6/(self.hidden_size+self.numLabels+1))),name="weights"))
            #self.bias = tf.Variable(tf.random_normal([1,self.fc1_hidden_size],mean=0,stddev=(np.sqrt(6/(self.hidden_size+self.numLabels+1))),name="bias"))

            self.fc1_output = tf.nn.relu(tf.matmul(lstm_layer_output, self.weights) + self.bias) # (None, fc1_hidden_size)
            

       # tf_idf... fc?
        with tf.variable_scope("tf_idf_fc"):
            
            self.weights2 = tf.Variable(tf.truncated_normal([self.numTfFeatures, self.fc2_hidden_size], stddev=0.1), name="fc2_w")
            self.bias2 = tf.Variable(tf.constant(0.1, shape=[1,self.fc2_hidden_size]), name="fc2_b")
            
            self.tf_idf_fc = tf.nn.relu(tf.matmul(self.tf_idf_x, self.weights2) + self.bias2) # (None, fc2_hidden_size)
            
       # tf_idf... fc?
        with tf.variable_scope("fc3"):

            self.fc3_input=tf.concat([self.fc1_output,self.tf_idf_fc],1) # (None, fc1_hidden_size + fc2_hidden_size)
            
            self.weights3 = tf.Variable(tf.truncated_normal([self.fc1_hidden_size+self.fc2_hidden_size, self.numLabels], stddev=0.1), name="fc3_w")
            self.bias3 = tf.Variable(tf.constant(0.1, shape=[1,self.numLabels]), name="fc3_b")
            
            self.logits = tf.matmul(self.fc3_input, self.weights3) + self.bias3 # (None, numLabels)
            
    
    def LossOP(self):
        self.cost_OP =  tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.place_y, logits=self.logits, name='costOp')
        self.loss =  tf.reduce_mean(self.cost_OP)
        

    def TrainOP(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
    
    def InitOP(self):
        self.init = tf.global_variables_initializer()
        
    def build(self):
        self.addPlaceholder()
        self.addVarWordEmbedding()
        self.LogitOP()
        self.LossOP()
        self.TrainOP()
        self.InitOP()
        
    def add_summary(self, sess):
        # tensorboard stuff
        ## Ops for vizualization
        # argmax(activation_OP, 1) gives the label our model thought was most likely
        # argmax(yGold, 1) is the correct label
        self.correct_predictions_OP = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.place_y,1))
        # False is 0 and True is 1, what was our average?
        self.accuracy_OP = tf.reduce_mean(tf.cast(self.correct_predictions_OP, "float"))
        # Summary op for regression output
        self.activation_summary_OP = tf.summary.histogram("output", self.logits)
        # Summary op for accuracy
        self.accuracy_summary_OP = tf.summary.scalar("accuracy", self.accuracy_OP)
        # Summary op for cost
        self.cost_summary_OP = tf.summary.scalar("cost", self.loss )
        # Merge all summaries
        self.all_summary_OPS = tf.summary.merge_all()
        # Summary writer
        self.file_writer = tf.summary.FileWriter(self.output_path, sess.graph)
    
    
    def train(self, sess , epoch):
        saver = tf.train.Saver()
        cost = 0
        diff = 1
        epoch_values=[]
        cost_values=[]
        acc_values=[]
        
        self.nbatches = 1000
        sess.run(self.init)
        self.add_summary(sess) 
        saver.save(sess, self.model_output)

        for i in range(epoch):

            if i > 1 and diff < .0001:
                print("change in cost %g; convergence."%diff)
                break
            else:
                rand_index=random.sample(range(0,len(self.train_data_x)),self.nbatches)
                
                fd={self.place_x:self.train_data_x[rand_index] ,self.place_y:self.train_data_y[rand_index], self.tf_idf_x : self.train_tfidf_x[rand_index] , self.lr : self.learning_rate}
                batch_seq_len=np.array([len(data) for data in self.train_data_x[rand_index]])
                fd={self.place_x:self.train_data_x[rand_index] ,self.place_y:self.train_data_y[rand_index],
                    self.lr : self.learning_rate, self.seq_len : batch_seq_len, self.dropout_:self.drop_out,
                    self.tf_idf_x : self.train_tfidf_x[rand_index] }
                # batch_seq_len=np.array([len(data) for data in self.train_data_x])
                # fd={self.place_x:self.train_data_x ,self.place_y:self.train_data_y,
                #     self.lr : self.learning_rate, self.seq_len : batch_seq_len, self.dropout_:self.drop_out,
                #     self.tf_idf_x : self.train_tfidf_x }
                sess.run([self.train_op],feed_dict=fd )

                # tensorboard
                if i % 20 == 0:
                    # Add epoch to epoch_values
                    summary_results, train_accuracy, newCost = sess.run(
                        [self.all_summary_OPS, self.accuracy_OP, self.loss], 
                        feed_dict=fd
                    )
                    # Generate accuracy stats on test data
                    epoch_values.append(i)
                    cost_values.append(newCost)
                    acc_values.append(train_accuracy)
                    diff = abs(newCost - cost)
                    cost = newCost
                    print("step %d, train accuracy %g"%(i, train_accuracy))
                    print("step %d, train cost %g"%(i, newCost))
                    print("step %d, change in cost %g"%(i, diff))


                    self.file_writer.add_summary(summary_results, i)
        
        ############################
        ### MAKE NEW PREDICTIONS ###
        ############################

        # Close tensorflow session
        sess.close()
        
    def testAcc(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            saver.restore(sess,  self.model_output)
            batch_seq_len=np.array([len(data) for data in self.test_data_x])
            fd={self.place_x:self.test_data_x , self.place_y:self.test_data_y, 
                self.lr : self.learning_rate, self.seq_len : batch_seq_len, self.dropout_: 0.0,
                self.tf_idf_x : self.test_tfidf_x }
            acc = sess.run([self.accuracy_OP],feed_dict=fd )
            print("- test acc {:04.2f}".format(100 * acc[0]))
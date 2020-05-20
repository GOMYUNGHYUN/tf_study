import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    def __init__(self, [seqLen, numLabels, numTfFeatures],[Train_X,Train_Y,Test_X,Test_Y, ,Val_X, Val_Y],gloveEmbedMat, [Train_tf_idf,Test_tf_idf,Val_tf_idf]):
        
        self.numLabels=numLabels
        self.seqLen=seqLen
        self.numTfFeatures=numTfFeatures
        
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
        
        self.hidden_size=50
        self.learning_rate=0.01
        
        self.output_path='data/temp2/LOG'
        self.model_output='data/temp2/MODEL'
    
    def addPlaceholder(self):
        # word_idx vect ~ [ None, sentence Number * length, numFeatures ]
        self.place_x = tf.placeholder(tf.float32, [None, None, self.numFeatures])
        
        self.seq_len= tf.placeholder(tf.int32, shape=[None])
        
        self.tf_idf_x = tf.placeholder(tf.float32, [None,self.numTfFeatures])
        
        self.place_y = tf.placeholder(tf.float32, [None, self.numLabels])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],name="lr")

    def addVarWordEmbedding(self):
        self.variable_word_embeddings = tf.Variable(self.embeddingMat, name="_word_embeddings", dtype=tf.float32, trainable=True)
        self.word_embeddings = tf.nn.embedding_lookup(self.variable_word_embeddings, self.place_x, name="word_embeddings")
        
    # use only embedding vect
    def LogitOP(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw, self.word_embeddings, self.seq_len,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output_lstm = tf.nn.dropout(output, dropout)
            output_lstm = tf.reshape(output_lstm, [-1, 2 * self.hidden_size])
        
        with tf.variable_scope("lstm_fc"):
            
            self.weights = tf.Variable(tf.random_normal([2 * self.hidden_size, self.numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6/(self.hidden_size+self.numLabels+1))),
                                       name="weights"))

            self.bias = tf.Variable(tf.random_normal([1,self.numLabels],
                                                mean=0,
                                                stddev=(np.sqrt(6/(self.hidden_size+self.numLabels+1))),
                                                name="bias"))
            
            self.logits = tf.matmul(output_lstm, W) + b
            self.logits = tf.reshape(self.logits, [-1, self.numLabels])
        
       # tf_idf... fc?
        with tf.variable_scope("fc"):
            W = tf.get_variable("w", shape=[2 * self.numTfFeatures, self.numLabels], dtype=tf.float32) # share variable

            b = tf.get_variable("b", shape=[self.numLabels], dtype=tf.float32, initializer=tf.zeros_initializer()) # share variable

            
            self.logits = tf.matmul(output_lstm, W) + b
            self.logits = tf.reshape(self.logits, [-1, self.numLabels])
    
            
        
            
    
    def LossOP(self):
        self.cost_OP = tf.nn.softmax_cross_entropy_with_logits(self.place_y, self.logits, axis=-1, name='loss')
        
    def TrainOP(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.cost_OP)
    
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
        self.cost_summary_OP = tf.summary.scalar("cost", self.cost_OP)
        
        # Merge all summaries
        self.all_summary_OPS = tf.summary.merge_all()
        # Summary writer
        self.file_writer = tf.summary.FileWriter(self.output_path, sess.graph)
    
    def addPadding(x):
        return x
    
    def train(self, sess , epoch):
        saver = tf.train.Saver()
        cost = 0
        diff = 1
        epoch_values=[]
        cost_values=[]
        acc_values=[]
        
        
        nbatches = 300
        sess.run(self.init)
        self.add_summary(sess) 
        saver.save(sess, self.model_output)

        for i in range(epoch):

            if i > 1 and diff < .0001:
                print("change in cost %g; convergence."%diff)
                break
            else:
                rand_index=random.sample(range(0,len(self.train_data_x)),nbatches)
                
                #fd={self.place_x:self.train_data_x[rand_index] ,self.place_y:self.train_data_y[rand_index], self.tf_idf_x : self.train_tfidf_x[rand_index] , self.lr : self.learning_rate}
                batch_seq_len=[len(data) for data in self.train_data_y[rand_index]]
                fd={self.place_x:self.train_data_x[rand_index] ,self.place_y:self.train_data_y[rand_index],self.lr : self.learning_rate, self.seq_len : batch_seq_len }
                train_loss= sess.run([self.training_OP],feed_dict=fd )

                # tensorboard
                if i % 50 == 0:
                    rand_index=random.sample(range(0,len(self.val_data_x)),nbatches)
                    batch_seq_len=[len(data) for data in self.val_data_y[rand_index]]
                    fd={self.place_x:self.val_data_x[rand_index] ,self.place_y:self.val_data_y[rand_index],self.lr : self.learning_rate, self.seq_len : batch_seq_len }
                    # Add epoch to epoch_values
                    summary_results, train_accuracy, newCost = sess.run(
                        [self.all_summary_OPS, self.accuracy_OP, self.cost_OP], 
                        feed_dict=fd
                    )
                    # Generate accuracy stats on test data
                    epoch_values.append(i)
                    cost_values.append(newCost)
                    acc_values.append(train_accuracy)
                    diff = abs(newCost - cost)
                    cost = newCost
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                    print("step %d, cost %g"%(i, newCost))
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
            fd={self.place_x:self.test_data_x ,self.place_y:self.test_data_y}
            acc = sess.run([self.accuracy_OP],feed_dict=fd )
            
            print("- test acc {:04.2f}".format(100 * acc[0]))
        

        
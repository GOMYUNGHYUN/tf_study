import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    def __init__(self, numFeatures, numLabels,Train_X,Train_Y,Test_X,Test_Y):
        self.numLabels=numLabels
        self.numFeatures=numFeatures
        self.train_data_x=Train_X
        self.train_data_y=Train_Y
        self.test_data_x=Test_X
        self.test_data_y=Test_Y
        self.output_path='data/temp/LOG'
        self.model_output='data/temp/MODEL'
    
    def addPlaceholder(self):
        self.place_x = tf.placeholder(tf.float32, [None, self.numFeatures])
        # yGold = Y-matrix / label-matrix / labels... This will be our correct answers
        # matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here 
        # means that we can hold any number of emails
        self.place_y = tf.placeholder(tf.float32, [None, self.numLabels])
        
        
    def addVariables(self):
        self.weights = tf.Variable(tf.random_normal([self.numFeatures,self.numLabels],
                                               mean=0,
                                               stddev=(np.sqrt(6/(self.numFeatures+self.numLabels+1))),
                                               name="weights"))

        self.bias = tf.Variable(tf.random_normal([1,self.numLabels],
                                            mean=0,
                                            stddev=(np.sqrt(6/(self.numFeatures+self.numLabels+1))),
                                            name="bias"))
        
        
    
    
    def LogitOP(self):
        print(self.place_x,self.place_y)
        print(self.weights, self.bias)
        self.apply_weights_OP = tf.matmul(self.place_x , self.weights, name="apply_weights_matmul")
        self.add_bias_OP = tf.add(self.apply_weights_OP, self.bias, name="add_bias") 
        self.activation_OP = tf.nn.sigmoid(self.add_bias_OP, name="activation")
        
    def LossOP(self):
        self.cost_OP = tf.nn.l2_loss(self.activation_OP-self.place_y, name="squared_error_cost")
        
    def TrainOP(self):
        self.learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step= 1,
                                              decay_steps=self.train_data_x.shape[0],
                                              decay_rate= 0.95,
                                              staircase=True)
            
        self.training_OP = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.cost_OP)
    
    def InitOP(self):
        self.init = tf.global_variables_initializer()
        
    def build(self):
        self.addPlaceholder()
        self.addVariables()
        self.LogitOP()
        self.LossOP()
        self.TrainOP()
        self.InitOP()
        
    def add_summary(self, sess):
        # tensorboard stuff
        ## Ops for vizualization
        # argmax(activation_OP, 1) gives the label our model thought was most likely
        # argmax(yGold, 1) is the correct label
        self.correct_predictions_OP = tf.equal(tf.argmax(self.activation_OP,1),tf.argmax(self.place_y,1))
        # False is 0 and True is 1, what was our average?
        self.accuracy_OP = tf.reduce_mean(tf.cast(self.correct_predictions_OP, "float"))
        # Summary op for regression output
        self.activation_summary_OP = tf.summary.histogram("output", self.activation_OP)
        # Summary op for accuracy
        self.accuracy_summary_OP = tf.summary.scalar("accuracy", self.accuracy_OP)
        # Summary op for cost
        self.cost_summary_OP = tf.summary.scalar("cost", self.cost_OP)
        # Summary ops to check how variables (W, b) are updating after each iteration
        self.weightSummary = tf.summary.histogram("weights", self.weights.eval(session=sess))
        self.biasSummary = tf.summary.histogram("biases", self.bias.eval(session=sess))
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
                fd={self.place_x:self.train_data_x[rand_index] ,self.place_y:self.train_data_y[rand_index]}
                train_loss= sess.run([self.training_OP],feed_dict=fd )

                # tensorboard
                if i % 10 == 0:
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
        

        
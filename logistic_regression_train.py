################
### PREAMBLE ###
################

from __future__ import division
import tensorflow as tf
import tarfile
import os
import matplotlib.pyplot as plt
import time


from mymodel import MyModel as baseModel


import numpy as np
import pandas as pd
import sys

from sklearn import preprocessing


vocabs_path='data/processed_data/vocab.txt'
glove_mat_path='data/processed_data/embedding_mat.npz'
tf_idf_vocabs_path='data/processed_data/tf_idf_vocab.txt'
tf_idf_mat_path='data/processed_data/tf_idf_mat.npz'
train_text_path='data/train_test/train.pkl'
test_text_path='data/train_test/test.pkl'


def load_vocab(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx
    # format -> dict[word] = mat index
    return d
def load_embedding_mat(filename):
    with np.load(filename) as data:
        return data["embeddings"]

def importProcessedData():
    global vocabs_path
    global glove_mat_path
    global vocabs_path
    global tf_idf_mat_path
    
    vocabs=load_vocab(vocabs_path)
    #  [ num vocabs x glove_dim ] mat
    glove_mat=load_embedding_mat(glove_mat_path)
    
    tf_idf_vocabs=load_vocab(vocabs_path)
    #  [ num all text x tf_idf_vocabs ] mat
    tf_idf_mat=load_embedding_mat(tf_idf_mat_path)
    
    return vocabs, glove_mat, tf_idf_vocabs, tf_idf_mat

def importTextData():
    global train_text_path
    global test_text_path

    train=pd.read_pickle(train_text_path)
    test=pd.read_pickle(test_text_path)
    
    return train,test

def sentenceToMeanVect(datasets,vocab,embeddingMat):
    # datasets -> multiple news
    # one news format : [ [ word1, word2 ...    ], [ word1, word2 ... ] ]
    global glove_dim
    new_datasets=np.zeros([len(datasets),glove_dim])
    for news_index,dataset in enumerate(datasets):
        one_news=np.zeros([len(dataset),glove_dim])
        for sentence_idx,words in enumerate(dataset):
            words_index_list=[]
            for w in words:
                if w in vocab.keys():
                    words_index_list.append(vocab[w])
            if len(words_index_list)!=0:
                one_news[sentence_idx]=embeddingMat[words_index_list].mean(axis=0)
        new_datasets[news_index]=one_news.mean(axis=0)
            
    return new_datasets
    
def getPlaceholder(numFeatures,numLabels):
    X = tf.placeholder(tf.float32, [None, numFeatures])
    # yGold = Y-matrix / label-matrix / labels... This will be our correct answers
    # matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here 
    # means that we can hold any number of emails
    yGold = tf.placeholder(tf.float32, [None, numLabels])
    
    return X,yGold

def getVariables(numFeatures,numLabels):
    weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                           mean=0,
                                           stddev=(np.sqrt(6/(numFeatures+
                                                             numLabels+1))),
                                           name="weights"))

    bias = tf.Variable(tf.random_normal([1,numLabels],
                                        mean=0,
                                        stddev=(np.sqrt(6/(numFeatures+numLabels+1))),
                                        name="bias"))
    return weights,bias
    
def setGraph(lr):
    

    
    ######################
    ### PREDICTION OPS ###
    ######################
    # PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
    apply_weights_OP = tf.matmul(X, w1, name="apply_weights")
    add_bias_OP = tf.add(apply_weights_OP, b1, name="add_bias") 
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")


    #####################
    ### EVALUATION OP ###
    #####################

    # COST FUNCTION i.e. MEAN SQUARED ERROR
    cost_OP = tf.nn.l2_loss(activation_OP-Y, name="squared_error_cost")


    #######################
    ### OPTIMIZATION OP ###
    #######################

    # OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
    training_OP = tf.train.GradientDescentOptimizer(lr).minimize(cost_OP)
    
    return training_OP


if __name__ == "__main__":
    train_text,test_text = importTextData()
    vocabs, glove_mat, tf_idf_vocabs, tf_idf_mat = importProcessedData()
    trainX,trainY,testX,testY = train_text.data, train_text.label, test_text.data, test_text.label

    # convert one hot encoding vector

    le = preprocessing.LabelEncoder()
    le.fit(trainY.unique()) # label  FAKE , REAL -> 0, 1 변경
    trainY=le.transform(trainY.tolist()).reshape([-1,1])
    testY=le.transform(testY.tolist()).reshape([-1,1])

    oe = preprocessing.OneHotEncoder() # one hot encoding
    oe.fit(trainY)
    trainY=oe.transform(trainY).toarray()
    testY=oe.transform(testY).toarray()


        
    # tf-idf vect data split to train, test data
    train_tf_idf_mat=tf_idf_mat[train_text.index]
    test_tf_idf_mat=tf_idf_mat[test_text.index]


    #########################
    ### GLOBAL PARAMETERS ###
    #########################
    glove_dim=100
    tf_idf_dim=tf_idf_mat.shape[1]
    numFeatures = glove_dim + tf_idf_dim
    numLabels = len(list(le.classes_)) #  list(le.classes_) => [ FAKE , REAL]
    
    mode=sys.argv[1]
    
    ## TRAINING SESSION PARAMETERS
    # number of times we iterate through training data
    # tensorboard shows that accuracy plateaus at ~25k epochs
    numEpochs = 500
    # a smarter learning rate for gradientOptimizer

    
    
    if mode=='base' :
        """
        
        base mode : sentence -> fixed vect
        
        """
        # data ~ ndarray format
        print('sentenceToMeanVect using Glove')
        trainX_vect=sentenceToMeanVect(trainX.tolist(),vocabs, glove_mat)
        testX_vect=sentenceToMeanVect(testX.tolist(),vocabs, glove_mat)
        
        print('glove vect concat tf_idf')
        trainX_vect = np.concatenate( (trainX_vect, train_tf_idf_mat), axis=1)
        testX_vect = np.concatenate( (testX_vect, test_tf_idf_mat), axis=1)
        
        
        #####################
        ### RUN THE GRAPH ###
        #####################

        # Create a tensorflow session
        sess = tf.Session()
        model=baseModel(numFeatures, numLabels,trainX_vect,trainY,testX_vect,testY)
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess: #config=tf.ConfigProto(log_device_placement=True)
            print('build')
            model.build()
            print('train')
            model.train(sess,numEpochs)
            
        model.testAcc()
        
    sys.exit(0)
    # To view tensorboard:
        #1. run: tensorboard --logdir=/path/to/log-directory
        #2. open your browser to http://gomyunghyun:6006/

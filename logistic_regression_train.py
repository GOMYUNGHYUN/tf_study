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

"""
훈련/평가 실행
python logistic_regression_train base

텐서보드 확인
tensorboard --logdir=./data/temp/LOG/


1. 전처리한 데이터 로드
2. tensor에 넣기 위한 데이터 처리
    1. feature 생성 
        1.text -> glove 임베딩 벡터로 표현하기( 벡터 mean )
        2.glove + tf_idf 벡터 합치기
    2. label 생성
        1. label encoding & one hot encoding
3. model 로드 & 훈련/시험
   (세부사항은 mymodel.py)

"""
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
    numEpochs = 3000
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

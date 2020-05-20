################
### PREAMBLE ###
################

from __future__ import division
import tensorflow as tf
import tarfile
import os
import matplotlib.pyplot as plt
import time




import pickle
import numpy as np
import pandas as pd
import sys

from sklearn import preprocessing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

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
vocabs_path='data/processed_data2/vocab.txt'
glove_mat_path='data/processed_data2/embedding_mat.npz'
tf_idf_vocabs_path='data/processed_data2/tf_idf_vocab.txt'
tf_idf_mat_path='data/processed_data2/tf_idf_mat.npz'

train_idx_path='data/common_data_idx/train.pkl'
test_idx_path='data/common_data_idx/test.pkl'
val_idx_path='data/common_data_idx/val.pkl'
raw_text_path='data/fake_or_real_news.csv'

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
    

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


def removeStopWords(x):
    global stop_words
    global stemmer
    res=[]
    for sentence in x:
        words=word_tokenize(sentence)
        words=filter(lambda x: x not in stop_words and x.isalpha(), words)
        words = [stemmer.stem(w) for w in words]
        res.append( [w.lower() for w in words] )
    return res

    
def importData():
    global raw_text_path
    print("loading text data")
    # format -> title, text, label
    df=pd.read_csv(raw_text_path,usecols=[1,2,3])
    # 문장 분리
    df.title=df.title.map(lambda x: [x])
    df.text=df.text.map(lambda x: [i.strip() for i in x.split('\n')])
    print('preprocess - lower, only alphabet, remove stop word')
    df['data']=df.title+df.text
    # 불용어 제거
    df['data']=df['data'].map(removeStopWords)
    
    return df.loc[:,['data','label']]

def importTextData():
    global train_idx_path
    global test_idx_path
    global val_idx_path
    
    df=importData()
    
    with open(train_idx_path, 'rb') as f:
        train=df.iloc[pickle.load(f)]
    with open(test_idx_path, 'rb') as f:
        test=df.iloc[pickle.load(f)]
    with open(val_idx_path, 'rb') as f:
        val=df.iloc[pickle.load(f)]
        
    return train,test,val

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

def getWord2IDvect(datasets,wordId_dict,maxSeqLen):
    newDatasets=[]
    for news_index,dataset in enumerate(datasets):
        sentenceIds=[]
        
        for sentence_idx,words in enumerate(dataset):
            sentenceIds.append(wordId_dict['bos'])
            words=words[:maxSeqLen]
            for word in words:
                if word in wordId_dict:
                    sentenceIds.append(wordId_dict[word])
                else:
                    word='unk'
                    sentenceIds.append(wordId_dict[word])
            sentenceIds.append(wordId_dict['eos'])
        newDatasets.append(sentenceIds)
    return newDatasets
    
    

if __name__ == "__main__":
    
    
    train_text,test_text,val_text = importTextData()
    vocabs, glove_mat, tf_idf_vocabs, tf_idf_mat = importProcessedData()
    trainX,trainY,testX,testY,valX,valY = train_text.data, train_text.label, test_text.data, test_text.label,val_text.data, val_text.label

    
    # convert one hot encoding vector

    le = preprocessing.LabelEncoder()
    le.fit(trainY.unique()) # label  FAKE , REAL -> 0, 1 변경
    trainY=le.transform(trainY.tolist()).reshape([-1,1])
    testY=le.transform(testY.tolist()).reshape([-1,1])
    valY=le.transform(valY.tolist()).reshape([-1,1])

    oe = preprocessing.OneHotEncoder() # one hot encoding
    oe.fit(trainY)
    trainY=oe.transform(trainY).toarray()
    testY=oe.transform(testY).toarray()
    valY=oe.transform(valY).toarray()
        
    # tf-idf vect data split to train, test data
    train_tf_idf_mat=tf_idf_mat[train_text.index]
    test_tf_idf_mat=tf_idf_mat[test_text.index]
    val_tf_idf_mat=tf_idf_mat[val_text.index]


    #########################
    ### GLOBAL PARAMETERS ###
    #########################

    
    mode=sys.argv[1]
    
    ## TRAINING SESSION PARAMETERS
    # number of times we iterate through training data
    # tensorboard shows that accuracy plateaus at ~25k epochs
    numEpochs = 1000
    seqLen=30
    # a smarter learning rate for gradientOptimizer

    
    
    if mode == 'base' :
        """
        base mode : sentence -> fixed vect
        """
        
        from mymodel import MyModel as baseModel
        
        # data ~ ndarray format
        print('sentenceToMeanVect using Glove')
        trainX_vect=sentenceToMeanVect(trainX.tolist(),vocabs, glove_mat)
        testX_vect=sentenceToMeanVect(testX.tolist(),vocabs, glove_mat)
        valX_vect=sentenceToMeanVect(valX.tolist(),vocabs, glove_mat)
        
        print('glove vect concat tf_idf')
        trainX_vect = np.concatenate( (trainX_vect, train_tf_idf_mat), axis=1)
        testX_vect = np.concatenate( (testX_vect, test_tf_idf_mat), axis=1)
        valX_vect = np.concatenate( (valX_vect, val_tf_idf_mat), axis=1)
        
        

        #parameter
        glove_dim=100
        tf_idf_dim=tf_idf_mat.shape[1]
        numFeatures = glove_dim + tf_idf_dim
        numLabels = len(list(le.classes_)) #  list(le.classes_) => [ FAKE , REAL]
        
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
        
    elif mode == 'LSTM' :
        
        """
        
        LSTM mode
        
        using embedding matxrix with unknown
        
        """
        numTfFeatures=train_tf_idf_mat.shape[1]
        
        #from myLstmModel import MyModel as lstmModel
        # get word id vector
        trainX_vect=getWord2IDvect(trainX,vocabs,seqLen)
        testX_vect=getWord2IDvect(testX,vocabs,seqLen)
        valX_vect=getWord2IDvect(valX,vocabs,seqLen)
        
        # text word -> index
        sess = tf.Session()
        model=lstmModel(
            [seqLen, numLabels,numTfFeatures],
            [trainX_vect,trainY,testX_vect,testY,valX_vect,valY]
            glove_mat,
            [train_tf_idf_mat, test_tf_idf_mat, val_tf_idf_mat]
        )
        
        
        
        
    sys.exit(0)
    # To view tensorboard:
        #1. run: tensorboard --logdir=/path/to/log-directory
        #2. open your browser to http://gomyunghyun:6006/

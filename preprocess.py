import numpy as np
import pandas as pd
import tensorflow as tf
import tarfile
import os
import sklearn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


"""
preprocess.py

	1. 뉴스 데이터 로드
		1. 소문자 변경
		2. 알파벳만 뽑기
		3. 불요어 제거
		4. 제목, 본문 합치기
	2. 뉴스 데이터 ~ 단어 토크 나이즈(문장은 그냥 split(\n)했음)
	3. glove 로드
	4. glove vocab과 뉴스 데이터 vocab 일치하는 것만 뽑음
	5. 위 vocab , embedding matrix 저장
	6. tf-idf 벡터 만들기
		1. sklearn TfidfVectorizer 사용
			1. 영어만 쓰기, 출현 단어 빈도 10 이상만 사용, 단어 200개만 사용
		2. text 벡터로 바꾸기
		3. 저장

"""
def save_vocab(vocab, filename):
    print("save vocab")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))
    
def save_embedding_matrix(vocabs,glove, filename, dim):
    embeddings = np.zeros([len(vocabs), dim])
    for k,v in glove.items():
        if k in vocabs:
            word_idx=vocabs.index(k)
            embeddings[word_idx]=v
    np.savez_compressed(filename, embeddings=embeddings)
    
def getTokens(datasets):
    # datasets -> title, text
    print("get all tokens")
    vocab_words = set()
    for one_news in datasets:
        for sentence_list in one_news:
            vocab_words.update(sentence_list)
            
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words


def removeStopWords(x):
    global stop_words
    res=[]
    for sentence in x:
        words=word_tokenize(sentence)
        words=filter(lambda x: x not in stop_words and x.isalpha(), words)
        res.append( [w.lower() for w in words] )
    return res

    
def importData():
    print("loading text data")
    # format -> title, text, label
    df=pd.read_csv('data/fake_or_real_news.csv',usecols=[1,2,3])
    # 문장 분리
    df.title=df.title.map(lambda x: [x])
    df.text=df.text.map(lambda x: [i.strip() for i in x.split('\n')])
    print('preprocess - lower, only alphabet, remove stop word')
    df['data']=df.title+df.text
    # 불용어 제거
    df['data']=df['data'].map(removeStopWords)
    
    return df.loc[:,['data','label']]


def getGlove(dim):
    # https://nlp.stanford.edu/projects/glove/
    # 6B tokens, 400K voca -> 60억 토큰 사용 & 40만 단어 사전
    df = pd.read_csv('data/glove.6B/glove.6B.%sd.txt' %(dim), sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}
    print("- done. {} tokens".format(len(list(glove.keys())))   )
    # format -> word : vector
    return glove


if __name__ == "__main__":
    #remove stop words
    stop_words = stopwords.words('english')
    text_df=importData()
    
    ## get glove vocab, vector
    print('load glove and make embedding matrix')
    glove_dim=100
    raw_glove=getGlove(glove_dim)
    
    # get inter tokens on Raw Dataset , Glove Vocab
    vocab=getTokens(text_df.data) & set(raw_glove.keys())
    vocab=list(vocab)
    # save tokens
    print('save embedding matrix')
    save_vocab(vocab,'data/processed_data/vocab.txt')
    save_embedding_matrix(vocab,raw_glove,'data/processed_data/embedding_mat.npz',glove_dim)
    
    #########################
    ## get tf-idf score
    print('get tf_idf matrix')
    tf_idf_dim=200
    text_format=text_df['data'].map(lambda x: ''.join([' '.join(ele) for ele in x])   ).tolist()
    vectorizer = TfidfVectorizer(stop_words='english',min_df=10,max_features=tf_idf_dim)
    tf_idf_mat = vectorizer.fit_transform(text_format).toarray()
    tf_idf_words=vectorizer.get_feature_names()
    print('save tf_idf matrix')
    save_vocab(tf_idf_words,'data/processed_data/tf_idf_vocab.txt')
    np.savez_compressed('data/processed_data/tf_idf_mat.npz', embeddings=tf_idf_mat)
    
    
    ##########
    ## split train / test & save
    print('split train / test & save')
    train, test = train_test_split(text_df, test_size=0.3)
    print('train size : %s , test size %s' % (train.shape[0],test.shape[0]) )
    train.to_pickle('data/train_test/train.pkl')
    test.to_pickle('data/train_test/test.pkl')
    
    
    
    
    
    
    
    
    
    
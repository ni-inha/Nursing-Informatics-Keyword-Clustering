#!/usr/bin/env python
# coding: utf-8

# # 라이브러리 불러오기

# In[1]:


import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from inflection import singularize
#import inflect
import re
import numpy as np
import pandas as pd
import sys
from pandas.core.arrays import StringArray
from tqdm import tqdm


# # 작업이 완료된 년도의 bi/tri/uni 읽고 저장

# - 2018년도

# In[2]:


csv_2018 = pd.read_csv('data/processed_data/ngram_data_frequency_2018.csv') #,encoding='cp949'
index = csv_2018['index_1'].dropna()
list = []
for i in range(len(index)):
    if(csv_2018['Flag_1'][i]== 0 and csv_2018['Flag_11'][i]=='X'):
        list.append(csv_2018['bigram/trigram'][i])
series = pd.Series(list)
bi_tri_dict_2018 = series.value_counts().index


# In[3]:


index = csv_2018['index_2'].dropna()
list = []
for i in range(len(index)):
    if(csv_2018['Flag_2'][i]== 0 and csv_2018['Flag_22'][i]=='X'):
        list.append(csv_2018['unigram'][i])
series = pd.Series(list)
uni_dict_2018 = series.value_counts().index


# - 2021년도

# In[4]:


csv_2021 = pd.read_csv('data/processed_data/ngram_data_frequency_2021.csv') #,encoding='cp949'
index = csv_2021['index_1'].dropna()
list = []
for i in range(len(index)):
    if(csv_2021['Flag_1'][i]=='X'):
        list.append(csv_2021['bigram/trigram'][i])
series = pd.Series(list)
bi_tri_dict_2021 = series.value_counts().index


# In[5]:


index = csv_2021['index_2'].dropna()
list = []
for i in range(len(index)):
    if(csv_2021['Flag_2'][i]=='X'):
        list.append(csv_2021['unigram'][i])
series = pd.Series(list)
uni_dict_2021 = series.value_counts().index


# # 작업할 년도의 데이터 불러오기

# In[ ]:


csv = pd.read_csv('data/raw_data/datalist_2012.csv',encoding='cp949') #,encoding='cp949'
abstract = csv['초록']
abstract_val = abstract.values


# - nan 값 제외하기

# In[ ]:


abstract_values = []
for i in range(len(abstract_val)):  #nan 값 제외하기
  if (abstract_val[i] == abstract_val[i]):
    abstract_values.append(abstract_val[i])


# - 특정 문자열 제외하기

# In[24]:


final_values = []
for i in range(len(abstract_values)):
 final_values.append(re.sub(r'[0-9]',"",abstract_values[i])) #https://engineer-mole.tistory.com/238 (문자열에서 특정 문자만 삭제하는 방법)
final_val = np.array(final_values)


# # 함수 정의

# In[ ]:


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
         
    return list
#https://www.geeksforgeeks.org/python-get-dictionary-keys-as-a-list/ 딕셔너리를 리스트로 받기


# In[ ]:


def replace(list):    
    result_list = []
    for i in range(len(list)):
      new_str = [s.replace(' ','_') for s in list[i]] #https://github.com/RaRe-Technologies/gensim/issues/388 문자열에 whitespace가 있으면 에러가 생김
      result_list.append(new_str)
        
    return result_list


# In[26]:


def replace_nf(list):
    new_str = [s.replace(' ','_') for s in list] #https://github.com/RaRe-Technologies/gensim/issues/388 문자열에 whitespace가 있으면 에러가 생김

    return new_str


# # ngram  생성

# In[27]:


stop_word = stopwords.words('english')
c_vec = CountVectorizer(stop_words=stop_word, ngram_range=(2,3)) #https://towardsdatascience.com/text-analysis-basics-in-python-443282942ec5 Text analysis with ngram


# In[28]:


ngrams = c_vec.fit_transform(final_val)
count_values = ngrams.toarray().sum(axis=0)
vocab = c_vec.vocabulary_


# In[29]:


len(vocab)


# In[30]:


ngrams = []
for i in range(len(final_val)):
  sent = []
  c_vec.fit_transform([final_val[i].astype('U')]) #여기에서 c_vec에 ngram화된 단어들이 입력됨, U로 유니코드 문자열로 전환해줌
  sent.append(c_vec.vocabulary_)
  ngrams.append(sent)


# In[31]:


results = []
for i in range(len(ngrams)):
  results.append(getList(ngrams[i][0])) #뒤에 [0]을 붙이면 딕셔너리의 키값이 읽힘 , 리스트에서 문자열로 바뀜


# # 유의어 대체

# In[32]:


synonym = pd.read_csv('data/processed_data/synonym.csv') #,encoding='cp949'
original = synonym['word']


# In[33]:


output = []
for i in range(len(results)):
    words = results[i]
    word_list = []
    for word in words:
        if(word in original):
            word = synonym.loc[synonym['word']==word,'replace']
            word_list.append(word)
        else:
            word_list.append(word)
    output.append(word_list)


# # 이전 년도의 단어 사전 생성

# In[34]:


bi_tri_dict = bi_tri_dict_2018.to_list() + bi_tri_dict_2021.to_list()
uni_dict = uni_dict_2018.to_list() + uni_dict_2021.to_list()


# - 제거 과정 수행

# In[35]:


result = []
for i in tqdm(range(len(output))):
    words = output[i]
    for word in words:
        if(word not in bi_tri_dict and word not in uni_dict):
            result.append(word)


# In[36]:


vocab_keyword = []
keyword = getList(vocab)
for word in keyword:
    if word in result:
        vocab_keyword.append(word)
    else:
        del vocab[word]


# In[37]:


len(vocab)


# # 단어를 df로 변환후 csv에 저장

# In[38]:


result_low = []
for word in vocab_keyword:
    result_low.append(singularize(word))
value = vocab.values()
vocab1 = dict(zip(result_low, value))


# In[39]:


df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab1.items()], reverse=True)).rename(columns={0: 'frequency', 1:'bigram/trigram'})


# In[40]:


df_ngram


# In[41]:


df_ngram.to_csv("data/processed_data/2012_processed_frequency_bi_tri.csv", mode='w')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


csv_2018 = pd.read_csv('data/processed_data/ngram_data_frequency_2018.csv') #,encoding='cp949'
index = csv_2018['index_1'].dropna()
list = []
for i in range(len(index)):
    if(csv_2018['Flag_1'][i]== 0 and csv_2018['Flag_11'][i]=='X'):
        list.append(csv_2018['bigram/trigram'][i])
series = pd.Series(list)
bi_tri_dict_2018 = series.value_counts().index


# In[4]:


index = csv_2018['index_2'].dropna()
list = []
for i in range(len(index)):
    if(csv_2018['Flag_2'][i]== 0 and csv_2018['Flag_22'][i]=='X'):
        list.append(csv_2018['unigram'][i])
series = pd.Series(list)
uni_dict_2018 = series.value_counts().index


# In[5]:


csv_2021 = pd.read_csv('data/processed_data/ngram_data_frequency_2021.csv') #,encoding='cp949'
index = csv_2021['index_1'].dropna()
list = []
for i in range(len(index)):
    if(csv_2021['Flag_1'][i]=='X'):
        list.append(csv_2021['bigram/trigram'][i])
series = pd.Series(list)
bi_tri_dict_2021 = series.value_counts().index


# In[6]:


index = csv_2021['index_2'].dropna()
list = []
for i in range(len(index)):
    if(csv_2021['Flag_2'][i]=='X'):
        list.append(csv_2021['unigram'][i])
series = pd.Series(list)
uni_dict_2021 = series.value_counts().index


# In[13]:


csv = pd.read_csv('data/raw_data/datalist_2016.csv',encoding='cp949') #,encoding='cp949'
abstract = csv['초록']
abstract_values = []
abstract_val = abstract.values
for i in range(len(abstract_val)):  #nan 값 제외하기
  if (abstract_val[i] == abstract_val[i]):
    abstract_values.append(abstract_val[i])

final_values = []
for i in range(len(abstract_values)):
 final_values.append(re.sub(r'[0-9]',"",abstract_values[i])) #https://engineer-mole.tistory.com/238 (문자열에서 특정 문자만 삭제하는 방법)


# In[14]:


final_val = np.array(final_values)


# In[15]:


stop_word = stopwords.words('english')


# In[16]:


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
         
    return list
#https://www.geeksforgeeks.org/python-get-dictionary-keys-as-a-list/ 딕셔너리를 리스트로 받기
def replace(list):    
    result_list = []
    for i in range(len(list)):
      new_str = [s.replace(' ','_') for s in list[i]] #https://github.com/RaRe-Technologies/gensim/issues/388 문자열에 whitespace가 있으면 에러가 생김
      result_list.append(new_str)
        
    return result_list

def replace_nf(list):
    new_str = [s.replace(' ','_') for s in list] #https://github.com/RaRe-Technologies/gensim/issues/388 문자열에 whitespace가 있으면 에러가 생김

    return new_str


# In[17]:


c_vec = CountVectorizer(stop_words=stop_word, ngram_range=(1,3)) #https://towardsdatascience.com/text-analysis-basics-in-python-443282942ec5 Text analysis with ngram


# In[18]:


ngrams = []
for i in range(len(final_val)):
  sent = []
  c_vec.fit_transform([final_val[i].astype('U')]) #여기에서 c_vec에 ngram화된 단어들이 입력됨, U로 유니코드 문자열로 전환해줌
  sent.append(c_vec.vocabulary_)
  ngrams.append(sent)


# In[19]:


results = []
for i in range(len(ngrams)):
  results.append(getList(ngrams[i][0])) #뒤에 [0]을 붙이면 딕셔너리의 키값이 읽힘 , 리스트에서 문자열로 바뀜


# In[28]:


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


# In[32]:


bi_tri_dict = bi_tri_dict_2018.to_list() + bi_tri_dict_2021.to_list()
uni_dict = uni_dict_2018.to_list() + uni_dict_2021.to_list()


# In[34]:


result = []
for i in tqdm(range(len(output))):
    words = output[i]
    word_list = []
    for word in words:
        if(word not in bi_tri_dict and word not in uni_dict):
            word_list.append(word)
    result.append(word_list)


# In[38]:


result_low = []
for i in range(len(result)):
    lowcase = []
    for word in result[i]:
     lowcase.append(singularize(word))
    result_low.append(lowcase)


# In[ ]:


preprocessed_sentences = replace(result_low)


# In[ ]:


preprocessed_sentences


# In[ ]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases


# In[ ]:


bigram_transformer = Phrases(preprocessed_sentences)
model = Word2Vec(bigram_transformer[preprocessed_sentences],window=5, min_count=1, workers=4, sg=0)


# In[ ]:


model.wv.save_word2vec_format('eng_phr2021_0404')


# In[ ]:


get_ipython().system('python -m gensim.scripts.word2vec2tensor --input eng_phr2021_0404 --output eng_phr2021_0404')


# In[ ]:





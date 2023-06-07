#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[40]:


csv = pd.read_csv('data/processed_data/ngram_data_frequency_2021.csv') #,encoding='cp949'
index = csv['index_1'].dropna()
list = []
for i in range(len(index)):
    if(csv['Flag_1'][i]=='O'):
        word = csv['bigram/trigram'][i].split(' ')
        for j in range(len(word)):
            list.append(word[j])
series = pd.Series(list)
bi_tri_dict = series.value_counts().index


# In[41]:


bi_tri_dict


# In[61]:


index = csv['index_2'].dropna()
list = []
for i in range(len(index)):
    if(csv['Flag_2'][i]=='O'):
        list.append(csv['unigram'][i])
series = pd.Series(list)
uni_dict = series.value_counts().index


# In[62]:


uni_dict


# In[53]:


words=[]
for i in range(len(csv['unigram'])):
    words.append(csv['unigram'][i])


# In[54]:


list = []
for word in words:
    if word in bi_tri_dict:
        continue
    else:
        list.append(word)


# In[56]:


len(list)


# In[ ]:





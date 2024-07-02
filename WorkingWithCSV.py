#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('example.csv')


# In[2]:


df


# # OPENING A CSV FILE FROM AN URL

# In[ ]:


import requests
from io import StringIO

url = ""
headers = {"User-Agent": "Mozilla/5.0 (Macinosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}
req = requests.get(url , headers=headers)
data= StringIO(req.text)

pd.read_csv(data)


# # SEP PARAMETER

# In[4]:


pd.read_csv('sample-2.tsv',sep='\t', names=['Nan','Annual budget tracker', 'NaN'])


# # Index_Col Parameter

# In[6]:


pd.read_csv('scores.csv',index_col='number')


# # HEADER PARAMETER

# In[10]:


pd.read_csv('friends.csv',header=1)


# # USE_COLS PARAMETER

# In[13]:


pd.read_csv('scores.csv', usecols=['days','gender','age','afftype'])


# # SQUEEZE PARAMETER

# In[17]:


import pandas as pd

# Read the CSV file without the 'squeeze' parameter
data = pd.read_csv('scores.csv', usecols=['gender'])

# Squeeze the DataFrame if it contains only one column
if len(data.columns) == 1:
    data = data.squeeze()

print(data)


# # SKIP ROWS / NROWS

# In[18]:


pd.read_csv('scores.csv',skiprows=[0,2])


# In[20]:


pd.read_csv('scores.csv',nrows=20)


# # ENOCDING PARAMETER

# pd.read_csv('zomato.csv', encoding='give the name same as the encoding of the file')

# # SKIP BAD LINES

# pd.read_csv('scores.csv',error_bad_lines=False)

# # DTYPE PARAMETER

# In[27]:


pd.read_csv('scores.csv',dtype={'days':float})


# # HANDLING DATES

# pd.read_csv('scores.csv',parse_dates=['date'])

# # CONVERTORS

# pd.read_csv('scores.csv',covertors={'column name':function name to be applied})

# # NA_VALUES PARAMETER

# In[31]:


pd.read_csv('scores.csv', na_values=[' '])


# # Loading a huge dataset in chunks

# In[34]:


dfs = pd.read_csv('scores.csv',chunksize=18)


# In[36]:


for chunks in dfs:
    print(chunks.shape)


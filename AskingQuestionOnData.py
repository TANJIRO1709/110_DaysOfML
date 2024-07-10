#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('friends.csv')


# # 1. How big is the data

# In[4]:


df.shape


# # 2. How does data look like

# In[5]:


df.head()


# # 3. How is the datatype of cols

# In[6]:


df.info()


# # 4. Are there missing values in data

# In[7]:


df.isnull().sum()


# # 5. HOW does the data look mathematically

# In[8]:


df.describe()


# # 6. Are there duplicate values

# In[9]:


df.duplicated().sum()


# # 7. What is the corelation between cols

# df.corr()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('cars.csv')


# In[3]:


df.sample(10)


# # OneHot Encoding using pandas

# In[4]:


df['brand'].value_counts()


# In[5]:


pd.get_dummies(df , columns=['fuel','owner'])


# # K-1 ENCODING

# In[6]:


pd.get_dummies(df , columns=['fuel','owner'] , drop_first= True)


# # Onehot Encoding using sklearn

# In[7]:


from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and the target variable is in the last column
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 0:4], df.iloc[:, -1], test_size=0.2)


# In[8]:


from sklearn.preprocessing import OneHotEncoder


# In[9]:


ohe = OneHotEncoder(drop='first',dtype=np.int32)


# In[10]:


X_train_new = ohe.fit_transform(X_train[['fuel','owner']]).toarray()
X_test_new = ohe.fit_transform(X_test[['fuel','owner']]).toarray()


# In[11]:


X_train_new


# In[12]:


np.hstack((X_train[['brand','km_driven']].values,X_train_new))


# # OneHot Encoding with top categories

# In[13]:


counts = df['brand'].value_counts()


# In[14]:


df['brand'].nunique()
threshold = 100


# In[15]:


repl = counts[counts<=threshold].index


# In[18]:


pd.get_dummies(df['brand'].replace(repl,'uncommon'),dtype=np.int32)


# In[ ]:





# In[ ]:





# In[ ]:





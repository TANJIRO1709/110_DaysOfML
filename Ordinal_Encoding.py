#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('customer.csv')
df.sample(10)


# In[3]:


df = df.iloc[:,2:]


# In[4]:


df.head()


# In[5]:


from sklearn.preprocessing import OrdinalEncoder


# In[6]:


from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and the target variable is in the last column
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:, -1], test_size=0.2)


# In[7]:


X_train


# In[8]:


oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])


# In[9]:


oe.fit(X_train)


# In[10]:


X_train = oe.transform(X_train)
X_test = oe.transform(X_test)


# In[11]:


X_train


# In[12]:


oe.categories_


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


le = LabelEncoder()


# In[15]:


le.fit(Y_train)


# In[16]:


le.classes_


# In[17]:


Y_train = le.transform(Y_train)
Y_test = le.transform(Y_test)


# In[18]:


Y_train


# In[ ]:





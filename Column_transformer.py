#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('customer.csv')


# In[3]:


df


# In[4]:


from sklearn.compose import ColumnTransformer


# In[5]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# In[6]:


from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and the target variable is 'purchased'
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(columns=['purchased']), df['purchased'], test_size=0.2)



# In[7]:


transformer = ColumnTransformer(transformers=[('tnf1',SimpleImputer(),['age']),('tnf2',OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']]),['review','education']),('tnf3',OneHotEncoder(sparse=False,drop='first'),['gender'])],remainder='passthrough')


# In[8]:


transformer.fit_transform(X_train)


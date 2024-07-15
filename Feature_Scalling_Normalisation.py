#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('wine_data (1).csv')
df.sample(15)


# In[4]:


sns.kdeplot(df['citric acid'])


# color_dict = {1:'red',2:'green',3:'blue'}
# sns.scatterplot(df['points'], df['price'], hue=df['Unnamed: 0'], palette=color_dict)
# 

# In[8]:


from sklearn.model_selection import train_test_split

if 'citric acid' in df.columns:

    X_train, X_test, Y_train, Y_test = train_test_split(df.drop('citric acid', axis=1), df['citric acid'], test_size=0.3, random_state=0)
    print(X_train.shape, X_test.shape)
else:
    print("The column 'points' does not exist in the dataframe.")


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[10]:


X_train_scaled = pd.DataFrame(X_train_scaled , columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled , columns=X_test.columns)


# In[11]:


np.round(X_train_scaled.describe(),1)


# In[12]:


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.scatter(X_train['volatile acidity'],X_train['pH'])
ax1.set_title("Before Scalling")
ax2.scatter(X_train_scaled['volatile acidity'],X_train_scaled['pH'], color='red')
ax2.set_title("After Scalling")
plt.show()


# In[13]:


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.set_title("Before Scalling")
sns.kdeplot(X_train['volatile acidity'],ax=ax1)
sns.kdeplot(X_train['pH'],ax=ax1)

ax2.set_title("After Scalling")
sns.kdeplot(X_train_scaled['volatile acidity'],ax=ax2)
sns.kdeplot(X_train_scaled['pH'],ax=ax2)


# # Robust scalling

# Xi' = Xi - Xmedian/(75-25)percentile values

# use while having a lot of outliers

# In[ ]:





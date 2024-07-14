#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Social_Network_Ads 2.csv')


# df=df.iloc[:,2:]

# In[3]:


df.sample(5)


# In[4]:


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(df.drop('Purchased', axis=1), df['Purchased'],test_size=0.3,random_state=0)

X_train.shape , X_test.shape


# # standard Scaller

# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


scaler.mean_


# In[8]:


X_train_scaled = pd.DataFrame(X_train_scaled , columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled , columns=X_test.columns)


# In[10]:


np.round(X_train_scaled.describe(),1)


# # Effect of scalling

# In[12]:


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.scatter(X_train['Age'],X_train['EstimatedSalary'])
ax1.set_title("Before Scalling")
ax2.scatter(X_train_scaled['Age'],X_train_scaled['EstimatedSalary'], color='red')
ax2.set_title("After Scalling")
plt.show()


# In[13]:


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.set_title("Before Scalling")
sns.kdeplot(X_train['Age'],ax=ax1)
sns.kdeplot(X_train['EstimatedSalary'],ax=ax1)

ax2.set_title("After Scalling")
sns.kdeplot(X_train_scaled['Age'],ax=ax2)
sns.kdeplot(X_train_scaled['EstimatedSalary'],ax=ax2)


# # Why scalling is important

# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


lr = LogisticRegression()
lr_scaled = LogisticRegression()


# In[17]:


lr.fit(X_train,Y_train)
lr_scaled.fit(X_train_scaled,Y_train)


# In[21]:


y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)


# In[22]:


from sklearn.metrics import accuracy_score


# In[24]:


print("Actual",accuracy_score(Y_test,y_pred) )
print("Scaled",accuracy_score(Y_test,y_pred_scaled) )


# In[25]:


df.describe()


# In[ ]:





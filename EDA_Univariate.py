#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('tested.csv')


# In[7]:


df.head()


# # 1. Categorical Data

# # CountPlot 

# In[4]:


import seaborn as sns


# In[10]:


#sns.countplot(df['Survived'])
df['Survived'].value_counts().plot(kind='bar')


# In[11]:


df['Pclass'].value_counts().plot(kind='bar')


# In[13]:


df['Embarked'].value_counts().plot(kind='bar')


# # Pie chart

# In[14]:


df['Survived'].value_counts().plot(kind='pie')


# In[15]:


df['Embarked'].value_counts().plot(kind='pie')


# In[16]:


df['Pclass'].value_counts().plot(kind='pie',autopct='%.2f')


# # 2. Numerical Data

# # a. Histogram

# In[26]:


import matplotlib.pyplot as plt
plt.hist(df['Fare'], bins=6)


# # b. distplot

# In[28]:


sns.distplot(df['Fare'])


# # c. Boxplot

# In[31]:


sns.boxplot(df['Age'])


# In[30]:


df['Age'].min()


# In[32]:


df['Age'].max()


# In[ ]:





# In[ ]:





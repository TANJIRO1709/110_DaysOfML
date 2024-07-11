#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[4]:


tips = sns.load_dataset('tips')


# In[5]:


tips


# In[6]:


titanic = pd.read_csv('tested.csv')


# In[8]:


flights = sns.load_dataset('flights')


# In[9]:


flights


# In[10]:


iris = sns.load_dataset('iris')


# In[11]:


iris


# # 1. Scatterplot(Numerical - Numerical)

# In[16]:


import matplotlib.pyplot as plt
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.show()


# In[19]:


sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)
plt.show()


# In[21]:


sns.scatterplot(x='total_bill', y='tip', hue='sex', style='smoker', size='size', data=tips)
plt.show()


# # 2. barplot ( Numerical - Categorical)

# In[22]:


titanic.head()


# In[24]:


sns.barplot(x='Pclass', y='Age', data=titanic)

# Show the plot
plt.show()


# In[25]:


sns.barplot(x='Pclass', y='Fare', data=titanic)

# Show the plot
plt.show()


# In[28]:


sns.barplot(x='Pclass', y='Age', hue='Sex', data=titanic)

# Show the plot
plt.show()


# # 3. Boxplot ( Numerical - Categorical)

# In[31]:


sns.boxplot(x='Sex', y='Age', data=titanic)

# Show the plot
plt.show()


# In[32]:


sns.boxplot(x='Sex', y='Age', hue='Survived', data=titanic)

# Show the plot
plt.show()


# # 4. Distplot ( Numerical - Categorical)

# In[36]:


sns.distplot(titanic[titanic['Survived']==0]['Age'],hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'],hist=False)


# # 5. HeatMap ( Categorical - Categorical)

# In[37]:


titanic.head()


# In[40]:


sns.heatmap(pd.crosstab(titanic['Pclass'],titanic['Survived']))


# In[48]:


titanic.groupby('Sex')['Survived'].mean()*100


# # 6.Clustermap  (Categorical - Categorical)

# In[51]:


sns.clustermap(pd.crosstab(titanic['Parch'],titanic['Survived']))


# # 7. Pairplot

# In[52]:


iris.head()


# In[53]:


sns.pairplot(iris)


# In[54]:


sns.pairplot(iris , hue ='species')


# # 8. Lineplot (Numerical - Numerical)

# In[55]:


flights.head()


# new = flights.groupby('year').sum().reset_index()

# sns.lineplot(new['year'],new['passenger'])

# In[66]:


sns.heatmap(flights.pivot_table(values='passengers',index='month',columns='year'))


# In[68]:


sns.clustermap(flights.pivot_table(values='passengers',index='month',columns='year'))


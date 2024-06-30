#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.read_csv('scores.csv')


# In[2]:


import numpy as np


# In[3]:


print("Hello World")


# In[4]:


myarry = np.array([12,36,45,12], np.int8)


# In[5]:


print(myarry)


# In[6]:


myarry


# In[7]:


myarry[2]


# In[8]:


myarry.shape


# In[9]:


myarry.dtype


# # Array Creation: From Numpy

# In[10]:


listarray = np.array([[1,2,3], [4,5,6],[10,15,18]])


# In[11]:


listarray


# In[12]:


listarray.dtype


# In[13]:


listarray.shape


# In[14]:


listarray.size


# In[15]:


np.array({32,64})


# In[16]:


zeroes = np.zeros((2,5))


# In[17]:


zeroes


# In[18]:


rng = np.arange(15)


# In[19]:


rng


# In[20]:


lspace = np.linspace(1,50,10)


# In[21]:


lspace


# In[22]:


emp = np.empty((4,6))


# In[23]:


emp


# In[24]:


ide = np.identity(45)


# In[25]:


ide


# In[26]:


arr = np.arange(99)


# In[27]:


arr


# In[28]:


arr.reshape(3,33)


# In[29]:


arr.ravel


# In[30]:


arr = arr.ravel()


# In[31]:


arr


# In[32]:


x = [[1,2,3],[4,5,6],[7,1,0]]


# In[33]:


ar = np.array(x)


# In[34]:


ar


# In[35]:


ar.sum(axis=0)


# In[36]:


ar.sum(axis=1)


# In[37]:


ar.T


# In[38]:


for item in ar.flat:
    print(item)


# In[39]:


ar.ndim


# In[40]:


ar.size


# In[41]:


ar.nbytes


# In[42]:


one = np.array([1,3,4,689,4])


# # Argmax gives max element index number

# In[43]:


one.argmax()


# In[44]:


one.argmin()


# In[45]:


one.argsort()


# In[46]:


ar


# In[47]:


ar.argsort()


# In[48]:


ar.argmin()


# In[49]:


ar.argsort(axis=0)


# In[50]:


ar.ravel()


# In[51]:


ar2 = np.array([[1,0,4],[2,1,3],[2,4,3]])


# In[52]:


ar+ar2


# In[53]:


ar*ar2


# In[54]:


np.sqrt(ar2)


# In[55]:


ar.sum()


# In[56]:


ar.max()


# In[57]:


ar.min()


# In[58]:


np.where(ar>5)


# In[59]:


import sys


# In[60]:


py_arr = [1,2,3,4]


# In[61]:


np_arr = np.array(py_arr)


# In[62]:


sys.getsizeof(1) * len(py_arr)


# In[63]:


np_arr.itemsize*np_arr.size


# # Pandas Notes

# In[64]:


import numpy as np
import pandas as pd


# In[65]:


dict1 = {
    "Names": ['sujal', 'ram','Rahul','Raj'],
    "marks": ['99','45','89','67'],
    "city": ['bbsr','mumbai','bangalore','Rourkela']
}


# In[66]:


df = pd.DataFrame(dict1)


# In[67]:


df


# In[68]:


df.to_csv('example.csv')


# In[69]:


df.to_csv('friends.csv', index=False)


# In[70]:


df.head(2)


# In[71]:


df.tail(2)


# In[72]:


df.describe()


# In[73]:


import pandas as pd


# In[74]:


score = pd.read_csv('scores.csv')


# In[75]:


score


# In[76]:


score['days']


# In[77]:


score['days'][45]


# In[78]:


score['days'][45] = 17


# In[79]:


score['days'][45]


# In[80]:


ser = pd.Series(np.random.rand(34))


# In[81]:


ser


# In[82]:


type(ser)


# In[85]:


newdf = pd.DataFrame(np.random.rand(334,5), index=np.arange(334))


# In[86]:


newdf


# In[87]:


newdf.head()


# In[88]:


type(newdf)


# In[89]:


newdf.describe()


# In[90]:


newdf.dtypes


# In[91]:


newdf[0][0] = "sujal"


# In[92]:


newdf.head()


# In[94]:


newdf.index


# In[96]:


newdf.columns


# In[97]:


newdf.to_numpy()


# In[98]:


newdf[0][0] = 0.2


# In[99]:


newdf.to_numpy()


# In[100]:


newdf.T


# In[101]:


newdf.head()


# In[103]:


newdf.sort_index(axis=0, ascending=False)


# In[104]:


newdf


# In[105]:


newdf.sort_index(axis=1, ascending=False)


# In[106]:


type(newdf[0])


# In[107]:


newdf.loc[0,0] = 654


# In[108]:


newdf.head(2)


# In[110]:


newdf.loc[[1,2],[2,3]]


# In[111]:


newdf.loc[:,[2,3]]


# In[113]:


newdf.loc[(newdf[0]<0.3)& (newdf[2]>0.3)]


# In[115]:


newdf.iloc[0,4]


# In[117]:


newdf.drop([0],axis=1)


# In[118]:


newdf[2].isnull()


# In[120]:


newdf


# In[121]:


newdf.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





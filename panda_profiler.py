#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('tested.csv')


# In[3]:


df.head()


# In[8]:


get_ipython().system('pip install pandas_profiling')


# # from pandas_profiling import ProfileReport
# # prof = ProfileReport(df)
# # prof.to_file(output_file='output.html')

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup


# headers = {'User-Agent':'Morzilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
# webpage = requests.get('https://www.ambitionbox.com/',headers=headers).text

# soup = BeautifulSoup(webpage, 'lxml')

# print(soup.pretify())

# for i in soup.find_all('h2'):
#     print(i.text.strip())

# soup.find_all('p', class_='rating')

# soup.find_all('a', class_='review-count')

# company = soup.find_all('a', class_='company-content-wrapper')

# name = []
# rating = []
# for i in company:
#     name.append(i.find('h2').text.strip())
#     rating.append(i.find('p',class_='ratiing')[0].text.strip())
#     
# d={'name':name, 'rating':'rating'}
# df = pd.DataFrame(d)
# 
# df

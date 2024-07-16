#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


# In[3]:


df = pd.read_csv('tested.csv')


# In[4]:


df.sample(10)


# # without pipelines

# In[5]:


df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2, random_state=42)


# In[7]:


X_train.head(2)


# In[8]:


df.isnull().sum()


# In[9]:


si_age = SimpleImputer()
si_Fare = SimpleImputer(strategy='most_frequent')
X_train_age = si_age.fit_transform(X_train[['Age']])
X_train_fare = si_Fare.fit_transform(X_train[['Fare']])
X_test_age = si_age.transform(X_test[['Age']])
X_test_fare = si_Fare.transform(X_test[['Fare']])


# In[10]:


X_train_age


# In[11]:


X_train_fare


# In[12]:


ohe_Sex = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_Embarked = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# In[13]:


X_train_sex = ohe_Sex.fit_transform(X_train[['Age']])
X_train_embarked = ohe_Embarked.fit_transform(X_train[['Fare']])
X_test_sex = ohe_Sex.transform(X_test[['Age']])
X_test_embarked = ohe_Embarked.transform(X_test[['Fare']])


# In[14]:


X_train_rem = X_train.drop(columns=['Sex','Fare','Age','Embarked'])


# In[15]:


X_test_rem = X_test.drop(columns=['Sex','Fare','Age','Embarked'])


# In[16]:


X_train_transformed = np.concatenate((X_train_rem,X_train_age,X_train_sex,X_train_embarked,X_train_fare),axis=1)
X_test_transformed = np.concatenate((X_test_rem,X_test_age,X_test_sex,X_test_embarked,X_test_fare),axis=1)


# In[17]:


X_train_transformed.shape


# In[18]:


clf = DecisionTreeClassifier()
clf.fit(X_train_transformed,Y_train)


# In[19]:


y_pred=clf.predict(X_test_transformed)


# In[20]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)


# In[21]:


import pickle


# In[22]:


import os

# Ensure the directory exists
os.makedirs('models', exist_ok=True)

# Save the objects
pickle.dump(ohe_Sex, open('models/ohe_Sex.pkl', 'wb'))
pickle.dump(ohe_Embarked, open('models/ohe_Embarked.pkl', 'wb'))
pickle.dump(clf, open('models/clf.pkl', 'wb'))


# In[23]:


ohe_Sex = pickle.load(open('models/ohe_Sex.pkl', 'rb'))
ohe_Embarked= pickle.load(open('models/ohe_Embarked.pkl', 'rb'))
clf = pickle.load(open('models/clf.pkl', 'rb'))


# In[24]:


test_input = np.array([2,'male',31.0,0,0,10.5,'S'],dtype=object).reshape(1,7)


# In[25]:


test_input


# In[26]:


test_input_sex = ohe_Sex.transform(test_input[:,1].reshape(1,1))


# In[27]:


test_input_Embarked = ohe_Embarked.transform(test_input[:,-1].reshape(1,1))


# In[28]:


test_input_age = test_input[:,2].reshape(1,1)


# In[29]:


import numpy as np

try:
    # Ensure all arrays are defined and have compatible dimensions
    test_input_transformed = np.concatenate(
        (test_input[:, [0, 3, 4, 5]], test_input_age, test_input_sex, test_input_Embarked), 
        axis=1
    )
except NameError as e:
    print(f"NameError: {e}")
except ValueError as e:
    print(f"ValueError: {e}")


# In[30]:


test_input_transformed.shape


# In[31]:


clf.predict(test_input_transformed)


# # Using pipelines

# In[32]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline , make_pipeline


# In[33]:


from sklearn.feature_selection import SelectKBest , chi2


# In[34]:


df.sample(10)


# In[35]:


X_train.head()


# In[36]:


Y_train.head()


# In[37]:


#simple imputer
trf1 = ColumnTransformer(
    [
        ('impute_Age', SimpleImputer(), [2]),
        ('impute_Fare', SimpleImputer(strategy='most_frequent'), [5])
    ],
    remainder='passthrough'
)


# In[38]:


#one hot encoding
trf2 = ColumnTransformer([
                 ('ohe_Sex_Embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6]) 
],
           
remainder='passthrough')


# In[39]:


#Scaling
trf3 =ColumnTransformer(
[('scale',MinMaxScaler(),slice(0,10))],
remainder='passthrough')


# In[40]:


# feature selection
trf4 = SelectKBest(score_func=chi2,k=8)


# In[41]:


#train the model
trf5 = DecisionTreeClassifier()


# # Creating Pipeline

# In[42]:


pipe = Pipeline([
    ('trf1',trf1),
     ('trf2',trf2),
     ('trf3',trf3),
     ('trf4',trf4),
     ('trf5',trf5)
])


# pipe.fit(X_train,Y_train)

# # Explore Pipeline

# In[49]:


from sklearn import set_config
set_config(display='diagram')


# y_pred1 = pipe.predict(X_test)

# In[50]:


accuracy_score(Y_test,y_pred)


# # GridSearch Using Pipeline

# In[51]:


params = {
    'trf5_max_depth':[1,2,3,4,5,None]
}


# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
# grid.fit(X_train, Y_train)
# 

# grid.best_score_

# grid.best_params_

# import pickle
# pickle.dump(pipe,open('pipe.pkl','wb'))

# pipe = pickle.load(open('pipe.pkl','rb'))

# test_input_2 = np.array([2,'male',31.0,0,0,10.5,'S'],dtype=object).reshape(1,7)

# pipe.predict(test_input_2 )

# In[ ]:





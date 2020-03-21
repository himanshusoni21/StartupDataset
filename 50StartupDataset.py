#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# In[ ]:





# In[4]:


df = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\50 Startups Dataset\\50_Startups.csv')
df.head()


# In[7]:


x = df.iloc[:,:-1]
y = df.iloc[:,4]


# In[12]:


#State feature is categorical type, so we need to perform one hot encoding
st = pd.get_dummies(x['State'],drop_first=True) 
st.head()


# In[22]:


#Now we need to drop column 'State' from the dataset and concat one hot encoded 'state' feature in dataset
x = x.drop('State',axis=1)


# In[24]:


x = pd.concat([x,st],axis=1)


# In[25]:


x.head()


# In[26]:


#Descriptive Statistics
x.describe()


# In[30]:


#split dataset into train data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[32]:


#Fitting machine learning Linear Regression Model to the training set 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)


# In[47]:


#predicting test set result
y_predict = reg.predict(x_test)
y_predict


# In[52]:


from sklearn.metrics import r2_score
sc = r2_score(y_test,y_predict)
sc


# In[ ]:


#R2 score is nearer to 1 ,means This model is good


# In[62]:


ax = plt.axes(projection='3d')
ax.scatter3D(x['R&D Spend'],x['Administration'],x['Marketing Spend'])
plt.xlabel('R&D Spend')
plt.ylabel('Administration Spend')
plt.show()


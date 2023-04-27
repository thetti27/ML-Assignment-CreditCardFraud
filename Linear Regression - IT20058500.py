#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# # Data Collection and Analysis

# In[7]:


ccfraud_dataset = pd.read_csv('C:\\Users\\User\\Desktop\\creditcard.csv')


# In[8]:


ccfraud_dataset.head()


# In[9]:


ccfraud_dataset.shape


# In[10]:


ccfraud_dataset.describe()


# In[11]:


ccfraud_dataset['Class'].value_counts()


# In[12]:


ccfraud_dataset.groupby('Class').mean()


# In[13]:


X = ccfraud_dataset.drop(columns='Class',axis=1)
Y = ccfraud_dataset['Class']


# In[14]:


print(X)


# In[15]:


print (Y)


# # Data Standardization

# In[16]:


scaler = StandardScaler()


# In[17]:


scaler.fit(X)


# In[18]:


standardized_data = scaler.transform(X)


# In[19]:


print(standardized_data)


# In[20]:


X = standardized_data
Y = ccfraud_dataset['Class']


# In[21]:


print(X)
print(Y)


# ## Splitting between training and test data

# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[98]:


sns.pairplot(ccfraud_dataset, x_vars=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28'], y_vars='Amount', height=4, aspect=1, kind='scatter')
plt.show()


# # Training the Model

# In[88]:


# Create a linear regression object
lr = LinearRegression()

# Fit the model to the training data
lr.fit(X_train, Y_train)


# In[89]:


# Predict the target variable for the test data
Y_pred = lr.predict(X_test)


# In[96]:


# Calculate the accuracy scores
mse_train = np.mean((Y_train - lr.predict(X_train)) ** 2)
mse_test = np.mean((Y_test - Y_pred) ** 2)
r2_train = lr.score(X_train, Y_train)
r2_test = lr.score(X_test, Y_test)

# Print the accuracy scores
print("Training set accuracy:")
print("MSE: ", mse_train)
print("R-squared score: ", r2_train)

print("Testing set accuracy:")
print("MSE: ", mse_test)
print("R-squared score: ", r2_test)


# In[ ]:





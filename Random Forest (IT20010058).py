#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# # Data Collection and Analysis

# In[2]:


#loading the diabetes dataset to a pandas Dataframe
creditcard_dataset = pd.read_csv('C:\\Users\\Hashini\\creditcard.csv')


# In[3]:


#Printing the first 5 rows of the dataset
creditcard_dataset.head()


# In[4]:


#Finding the size of the dataset
creditcard_dataset.shape


# In[5]:


#Getting the statistical measure of the data
creditcard_dataset.describe()


# In[6]:


# Seperation of types of transactions 0 > Legitimate 1> Fraudulent
creditcard_dataset['Class'].value_counts()


# In[7]:


#All features means based on the class
creditcard_dataset.groupby('Class').mean()


# In[8]:


#seperating the data and labels
X = creditcard_dataset.drop(columns='Class',axis=1)
Y = creditcard_dataset['Class']


# In[9]:


print (X)


# In[10]:


print (Y)


# # Data Standardization
# 

# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(X)


# In[13]:


standardized_data=scaler.transform(X) 


# In[14]:


print(standardized_data)


# In[15]:


X = standardized_data
Y = creditcard_dataset['Class']


# In[16]:


print(X)
print(Y)


# # Splitting between training and test data
# 

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X ,Y ,test_size = 0.2, stratify=Y, random_state=2)


# In[18]:


print(X.shape, X_train.shape, X_test.shape)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Train a random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions on the validation set
Y_pred = rf_model.predict(X_test)
Y2_pred = rf_model.predict(X_train)

# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
print('Validation accuracy of test data set:', accuracy)

accuracy = accuracy_score(Y_train, Y2_pred)
print('Validation accuracy of train data set:', accuracy)


# In[22]:


# Load the iris dataset
iris = load_iris()

# Train a random forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(iris.data, iris.target)

# Plot the first tree
plt.figure(figsize=(10,10))
plot_tree(rf_model.estimators_[0], filled=True, feature_names=iris.feature_names)
plt.show()


# In[ ]:





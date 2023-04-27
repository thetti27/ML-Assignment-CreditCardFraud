#!/usr/bin/env python
# coding: utf-8

# In[139]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 


# # Data Collection and Analysis

# In[140]:


#loading the diabetes dataset to a pandas Dataframe
creditcard_dataset = pd.read_csv('C:\\Users\\Hashini\\creditcard.csv')


# In[141]:


#Printing the first 5 rows of the dataset
creditcard_dataset.head()


# In[142]:


#Finding the size of the dataset
creditcard_dataset.shape


# In[143]:


#Getting the statistical measure of the data
creditcard_dataset.describe()


# In[144]:


# Seperation of types of transactions 0 > Legitimate 1> Fraudulent
creditcard_dataset['Class'].value_counts()


# In[145]:


#All features means based on the class
creditcard_dataset.groupby('Class').mean()


# In[146]:


#seperating the data and labels
X = creditcard_dataset.drop(columns='Class',axis=1)
Y = creditcard_dataset['Class']


# In[147]:


print (X)


# In[148]:


print (Y)


# # Data Standardization
# 

# In[149]:


scaler = StandardScaler()


# In[150]:


scaler.fit(X)


# In[151]:


standardized_data=scaler.transform(X) 


# In[152]:


print(standardized_data)


# In[153]:


X = standardized_data
Y = creditcard_dataset['Class']


# In[154]:


print(X)
print(Y)


# # Splitting between training and test data
# 

# In[155]:


X_train, X_test, Y_train, Y_test = train_test_split(X ,Y ,test_size = 0.2, stratify=Y, random_state=2)


# In[156]:


print(X.shape, X_train.shape, X_test.shape)


# In[157]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Train a random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions on the validation set
Y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
print('Validation accuracy:', accuracy)


# In[158]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()

# Train a random forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(iris.data, iris.target)

# Plot the first tree
plt.figure(figsize=(10,10))
plot_tree(rf_model.estimators_[0], filled=True, feature_names=iris.feature_names)
plt.show()


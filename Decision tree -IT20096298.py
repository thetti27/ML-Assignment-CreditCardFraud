#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from IPython.display import Image
from graphviz import Source


# # Data Collection and Analysis

# In[28]:


#loading the diabetes dataset to a pandas Dataframe
creditcard_dataset = pd.read_csv('D:\\lectures\\4th YEAR\\ML\\Assignment\\Datasets\\data\\creditcard.csv')


# In[29]:


#Printing the first 5 rows of the dataset
creditcard_dataset.head()


# In[30]:


#Finding the size of the dataset
creditcard_dataset.shape


# In[31]:


#Getting the statistical measure of the data
creditcard_dataset.describe()


# In[32]:


# Seperation of types of transactions 0 > Legitimate 1> Fraudulent
creditcard_dataset['Class'].value_counts()


# In[33]:


#All features means based on the class
creditcard_dataset.groupby('Class').mean()


# In[34]:


#seperating the data and labels
X = creditcard_dataset.drop(columns='Class',axis=1)
Y = creditcard_dataset['Class']


# In[35]:


print (X)


# In[36]:


print (Y)


# # Data Standardization
# 

# In[37]:


scaler = StandardScaler()


# In[38]:


scaler.fit(X)


# In[39]:


standardized_data=scaler.transform(X) 


# In[40]:


print(standardized_data)


# In[41]:


X = standardized_data
Y = creditcard_dataset['Class']


# # print(X)
# print(Y)

# # Splitting between training and test data
# 

# In[42]:


X_train, X_test, Y_train, Y_test = train_test_split(X ,Y ,test_size = 0.3, stratify=Y, random_state=42)


# In[43]:


print(X.shape, X_train.shape, X_test.shape)


# # Feature Handling for decision tree algorithm
# 

# In[44]:


X = creditcard_dataset.drop(columns=['Class'])
y = creditcard_dataset['Class']
# Handle missing values, categorical variables, etc.


# # Train the decision tree classifier using the training set

# In[45]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)


# # Use regularization

# In[46]:


dtc = DecisionTreeClassifier(ccp_alpha=0.01)


# In[47]:


dtc.fit(X_train, Y_train)


# # Use the trained model to make predictions

# In[48]:


train_accuracy = dtc.score(X_train, Y_train)
print("Accuracy on training set: {:.10f}".format(train_accuracy))


# In[49]:


Y_pred = dtc.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)


# In[50]:


print("Accuracy on test set: {:.10f}".format(accuracy))


# # Show the diagram 

# In[52]:


iris = load_iris()
X = iris.data
Y = iris.target

# Train the decision tree classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X, Y)

# Convert the feature names to a list
feature_names = list(iris.feature_names)

# Convert the NumPy array to a pandas DataFrame
X_df = pd.DataFrame(X, columns=feature_names)

# Export the decision tree as a DOT file
dot_data = export_graphviz(dtc, out_file=None, feature_names=feature_names)

# Use Graphviz to visualize the DOT file
graph = Source(dot_data)
graph.format = 'png'
Image(graph.render())


# In[ ]:





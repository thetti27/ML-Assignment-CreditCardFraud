#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Matplotlib library to plot the charts
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Library for the statistic data vizualisation
import seaborn

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Collection and Analysis

# In[2]:


#loading the diabetes dataset to a pandas Dataframe
creditcard_dataset = pd.read_csv('C:\\Users\\HP\\Desktop\\ML\\creditcard.csv')

df = pd.DataFrame(creditcard_dataset)


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


# # Training the Model
# 

# In[19]:


classifier = svm.SVC(kernel='sigmoid')


# ## Training the SVM Classifier
# 

# In[20]:


classifier.fit(X_train,Y_train)


# ## Model Evaluation

# ### Accuracy score
# 

# In[21]:


# Accuracy score on the training data 
X_train_predication = classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_predication,Y_train)


# In[22]:


print('Accuracy score of the training data:', training_data_accuracy)


# In[23]:


# Accuracy score on the test data 
X_test_predication = classifier.predict(X_test)
test_data_accuracy= accuracy_score(X_test_predication,Y_test)


# In[24]:


print('Accuracy score of the test data:', test_data_accuracy)


# # Creating a predictor

# In[25]:


input_data = (-0.273, -1.147, -0.621, -0.354, -0.732, -0.131, -0.344, -0.838, -0.076, -0.168, -0.026, 1.688, -0.929, -0.878, -0.68, 0.529, 0.804, -0.611, -0.163, -0.792, 0.424, 0.703, 0.26, -0.166, -0.078, 0.556, -0.34, -0.186, -0.847, -0.155)


# changing the imput data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)


if (prediction[0] == 0):
    print('The transaction is legitimate')
else:
    print('The transaction is fradulent')


# # Data Visualization

# In[27]:


df = pd.DataFrame(creditcard_dataset) # Converting data to Panda DataFrame


# In[29]:


# Description of statistic features (Sum, Average, Variance, minimum, 1st quartile, 2nd quartile, 3rd Quartile and Maximum)
df.describe() 


# ## Creating a Scatter Plot of fraud amounds against time

# In[31]:


#Creating 
df_fraud = df[df['Class'] == 1] # Recovery of fraud data
plt.figure(figsize=(15,10))
plt.scatter(df_fraud['Time'], df_fraud['Amount']) # Display fraud amounts according to their time
plt.title('Scatter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,175000])
plt.ylim([0,2500])
plt.show()


# ## Distribution of fraud to no fraud
# 

# In[35]:


number_fraud = len(creditcard_dataset[creditcard_dataset.Class == 1])
number_legit = len(creditcard_dataset[creditcard_dataset.Class == 0])

print('There are  '+ str(number_fraud) + ' frauds in the original dataset')
print('There are  '+ str(number_legit) + ' legitimate transactions in the original dataset')


# Due to this uneven distribution we can explain the extremely high accuractly of the predicted values. Our next aim is to create a more representative model. 

# # Improving the Model

# In[36]:


# Data seperation 

# Training dataset
df_train_all = df[0:150000] # Split the original dataset in two.
df_train_1 = df_train_all[df_train_all['Class'] == 1] #Seperating of the fraud and no fraud data
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('In this dataset, we have ' + str(len(df_train_1)) +" frauds so we need to take a similar number of legitimate")

df_sample=df_train_0.sample(300)
df_train = df_train_1.append(df_sample) # Gather the frouds with the legitinate tranactions 
df_train = df_train.sample(frac=1) # Nix the dataset


# In[37]:


X_train = df_train.drop(['Time', 'Class'],axis=1) # Drop the time and class features
y_train = df_train['Class'] # Create label
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


# In[38]:


#

df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'],axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)


# ## Ranking the features based on their correlation to the class 

# In[44]:




df_corr = df.corr() # Calculation of the correlation coefficients in pairs
rank = df_corr['Class'] # Retrieving the correlation coefficients per feature in relation to the feature class
df_rank = pd.DataFrame(rank) 
df_rank = np.abs(df_rank).sort_values(by='Class',ascending=False) # Ranking the absolute values of the coefficients
                                                                  # in descending order
df_rank.dropna(inplace=True) # Removing Missing Data (not a number)


# In[42]:


#Defining the new training dataset with only the most important features 

X_train_rank = df_train[df_rank.index[1:11]] # Select only the first 10 ranked features
X_train_rank = np.asarray(X_train_rank)


# In[43]:


#Defining the new testing dataset 

X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)


# ## Confusion Matrix

# In[55]:


from sklearn.metrics import confusion_matrix
import itertools


# In[56]:


class_names=np.array(['0','1']) # Binary label, Class = 1 (fraud) and Class = 0 (legitimate)


# In[57]:


# Function to plot the confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Model Reselection

# In[58]:


classifier = svm.SVC(kernel='linear')


# In[59]:


classifier.fit(X_train, y_train) #Training the model with the balanced dataset


# ## Testing the Model 

# In[60]:


prediction_SVM_all = classifier.predict(X_test_all)


# In[61]:


cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm,class_names)


# In[65]:


print('Our new model gives the accuracy proportion of ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[66]:


print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("the accuracy is : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[ ]:





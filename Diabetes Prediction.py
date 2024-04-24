#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:


diabetes_dataset = pd.read_csv(r"C:\Users\VEDANSHI\OneDrive\Desktop\diabetes.csv") 


# In[3]:


diabetes_dataset.head()


# In[4]:


diabetes_dataset.shape


# In[5]:


diabetes_dataset.describe()


# In[27]:


cols=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
for label in cols[:-1]:
   plt.hist(diabetes_dataset[diabetes_dataset["Outcome"]==1][label] ,color='blue',label='diabetic',alpha=0.7,density=True)
   plt.hist(diabetes_dataset[diabetes_dataset["Outcome"]==0][label],color='red' ,label='non-diabetic',alpha=0.7,density=True)
   plt.title(label)
   plt.ylabel("probability")
   plt.xlabel(label)
   plt.legend()
   plt.show()


# In[6]:


diabetes_dataset['Outcome'].value_counts()


# In[7]:


diabetes_dataset.groupby('Outcome').mean()


# In[8]:


X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# Standardizing the  data to bring everything on the common scale with mean=0 & SD=1

# In[9]:


scaler = StandardScaler()


# In[10]:


scaler.fit(X)


# In[11]:


standardized_data = scaler.transform(X)


# In[12]:


print(standardized_data)


# In[13]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[14]:


print(X)
print(Y)


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[16]:


print(X.shape, X_train.shape, X_test.shape)


# Putting a SVM model on the training data and making prediction
# here kernel will be linear polynomial function=(a*b+r)^d

# In[17]:


classifier = svm.SVC(kernel='linear')


# In[18]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[19]:


# making prediction & calculating accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[20]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[22]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[23]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Creating a Predictive model which can predict correct outcome with new set data

# In[24]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Prediction using Decision Tree Algorithm
# 
# > This project is concerned with creating a decision tree classifier, tuning its parameters, and visualizing it graphically.
# 

# In[1]:


# Importing libraries in Python
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
# Prepare the data data
iris = datasets.load_iris()
X = iris.data
y = iris.target


# # Describe the data

# In[36]:


# Describe the data
df = pd.DataFrame(X)
df['target'] = y
df.describe()


# In[82]:


#print out a few lines of data.
df.head()


# In[81]:


# checking the data shape (we have 150 observations with 5 features)
df.shape


# In[37]:


# inspect data types and look for missing values (no missing values)
df.info()


# In[83]:


# splitting data into train and test

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=10)


# ### Defining the Decision Tree Algorithm

# In[79]:


# Fit the classifier with default hyper-parameters
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test) 
acc1 = accuracy_score(y_test, y_pred1)
print ("Overall Decision tree accuracy: ",acc1)


# In[84]:


# Fit the classifier after tuning hyper-parameters

clf2 = DecisionTreeClassifier(splitter = 'random', random_state=0)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test) 
acc = accuracy_score(y_test, y_pred)
print ("Overall Decision tree accuracy: ",acc)
print(confusion_matrix(y_test, y_pred))


# In[91]:


# Text representation of classifier 1 (Decision tree with default hyper-parameters)
text_representation = tree.export_text(clf1)
print(text_representation)


# In[92]:


# Text representation of classifier 2 (Decision tree after tuning hyper-parameters)

text_representation = tree.export_text(clf2)
print(text_representation)


# ### Visualizing the Decision Tree 
# 

# In[85]:


#Plot Tree of classifier 1 with plot_tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf1, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)


# In[5]:


fig.savefig("decistion_tree.png")


# In[88]:


#Plot Tree of classifier 2 with plot_tree
fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(clf2, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)


# In[6]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


# In[95]:


#Plot Tree of classifier 1 with dtreeviz

from dtreeviz.trees import dtreeviz 

viz = dtreeviz(clf1, X, y,
                target_name="target",
                feature_names=iris.feature_names,
                class_names=list(iris.target_names))

viz


# In[90]:


#Plot Tree of classifier 2 with dtreeviz

from dtreeviz.trees import dtreeviz

viz = dtreeviz(clf2, X, y,
                target_name="target",
                feature_names=iris.feature_names,
                class_names=list(iris.target_names))

viz


# **You can now feed any new/test data to this classifer 2 after parameter tuning and it would be able to predict the right class accordingly.**

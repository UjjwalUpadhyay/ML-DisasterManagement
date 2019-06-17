#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


train_df = pd.read_csv('./AnimalMovement.csv')
test_df = pd.read_csv('./AnimalMovement.csv')
combine=[train_df, test_df]


# In[3]:


train_df.head()


# In[4]:


train_df.info()


# In[5]:


test_df.info()


# In[6]:


train_df.describe()


# In[7]:


g = sns.FacetGrid(train_df, col='DisasterPrediction')
g.map(plt.hist, 'Tensed', bins=20)


# In[8]:


g = sns.FacetGrid(train_df, col='DisasterPrediction')
g.map(plt.hist, 'DistanceMoved', bins=2)


# In[9]:


g = sns.FacetGrid(train_df, col='DisasterPrediction')
g.map(plt.hist, 'AltitudeMoved', bins=2)


# In[10]:


grid = sns.FacetGrid(train_df, col='DisasterPrediction', row='Tensed', height=2.2, aspect=1.6)
grid.map(plt.hist, 'AltitudeMoved', alpha=.5, bins=20)
grid.add_legend();


# In[11]:


X_train = train_df.drop(["DisasterPrediction", "Species", "SiteID", "DateTime", "CaptureEventID"], axis=1)
Y_train = train_df["DisasterPrediction"]
X_test  = test_df.drop(["DisasterPrediction","Species", "SiteID", "DateTime", "CaptureEventID"], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[12]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[13]:


from sklearn.ensemble import RandomForestClassifier as RFC
rfc_b = RFC()
rfc_b.fit(X_train,Y_train)
Y_pred = rfc_b.predict(X_train)
rfc_knn = round(rfc_b.score(X_train, Y_train) * 100, 2)
rfc_knn


# In[14]:


submission = pd.DataFrame({
        "CaptureEventID": test_df["CaptureEventID"],
        "DateTime": test_df["DateTime"],
        "Species": test_df["Species"],
        "AltitudeMoved": test_df["AltitudeMoved"],
        "DistanceMoved": test_df["DistanceMoved"],
        "Tensed": test_df["Tensed"],
        "DisasterPrediction": test_df["DisasterPrediction"]
    })
submission.to_csv('results_new.csv', index=False)


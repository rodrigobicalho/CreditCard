# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:51:51 2017

@author: Rodrigo
"""

#MNIST DATASET TEST
# Adjusting the data for Biased sample

import pandas as pd
import numpy as np
import csv as csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

# Reading te file
path = 'C:\\Users\\Rodrigo\\Documents\\Curso Data Science Big Data\\Projects\\Credit Card Default\\data.csv'


data = pd.read_csv(path, sep=";")

#Shuffling the data order
data = data.drop(data.index[0])
data = shuffle(data)

#Rearranging for Y to stay in the first column
labels =  list(data.columns.values)
data = data[['Unnamed: 0','Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']]



#Converting data to integer
data.X1 = data.X1.astype(int)
data.X2 = data.X2.astype(int)
data.X3 = data.X3.astype(int)
data.X4 = data.X4.astype(int)
data.X5 = data.X5.astype(int)/20
data.X6 = data.X6.astype(int)
data.X7 = data.X7.astype(int)
data.X8 = data.X8.astype(int)
data.X9 = data.X9.astype(int)
data.X10 = data.X10.astype(int)
data.X11 = data.X11.astype(int)
data.X12 = data.X12.astype(int)
data.X13 = data.X13.astype(int)
data.X14 = data.X14.astype(int)
data.X15 = data.X15.astype(int)
data.X16 = data.X16.astype(int)
data.X17 = data.X17.astype(int)
data.X18 = data.X18.astype(int)
data.X19 = data.X19.astype(int)
data.X20 = data.X20.astype(int)
data.X21 = data.X21.astype(int)
data.X22 = data.X22.astype(int)
data.X23 = data.X23.astype(int)
data.Y = data.Y.astype(int)

# Creating dummy variables
# Setting Sex to be 0 for female and 1 for male
data["X2"] = data["X2"].apply(lambda x:0 if x == 2 else 1)


# Education previous: 1 Graduate 2 University 3 High School , Others
data["X24"] = data["X3"].apply(lambda x:1 if x == 1 else 0)
data["X25"] = data["X3"].apply(lambda x:1 if x == 2 else 0)
data["X3"] = data["X3"].apply(lambda x:1 if x == 3 else 0)

#Marital Status 1: Married 2 Single 3. Others
data["X26"] = data["X4"].apply(lambda x:1 if x == 1 else 0)
data["X4"] = data["X4"].apply(lambda x:1 if x == 2 else 0)

# Substitute variable "Ammount previous payment" for % bull statement payed
#Ammount due - Payment 

for i in range(12,18):
    data["X"+str(i+6)] = data["X"+str(i)] - data["X"+str(i+6)]
    data["X"+str(i)] = data["X"+str(i)]/data["X1"]
    data["X"+str(i+6)] = data["X"+str(i+6)]/data["X1"]
    

# analysis of some features (variables X6 - X11)
print(data[["X6", "Y"]].groupby(['X6'], as_index=False).mean().sort_values(by='X6', ascending=False))
print(data[["X7", "Y"]].groupby(['X7'], as_index=False).mean().sort_values(by='X7', ascending=False))
print(data[["X8", "Y"]].groupby(['X8'], as_index=False).mean().sort_values(by='X8', ascending=False))
print(data[["X9", "Y"]].groupby(['X9'], as_index=False).mean().sort_values(by='X9', ascending=False))
print(data[["X10", "Y"]].groupby(['X10'], as_index=False).mean().sort_values(by='X10', ascending=False))
print(data[["X11", "Y"]].groupby(['X11'], as_index=False).mean().sort_values(by='X11', ascending=False))
# Based on this data, the ideal division is in 3 dummies: a) for -2 to 0 b) -1 c) Others
data["X27"] = data["X6"].apply(lambda x:1 if x < 1 else 0)
data["X28"] = data["X6"].apply(lambda x:1 if x == 1 else 0)

data["X29"] = data["X7"].apply(lambda x:1 if x == 2 else 0)
data["X30"] = data["X7"].apply(lambda x:1 if x == 2 else 0)

data["X31"] = data["X8"].apply(lambda x:1 if x == 2 else 0)
data["X32"] = data["X8"].apply(lambda x:1 if x == 2 else 0)

data["X33"] = data["X9"].apply(lambda x:1 if x == 2 else 0)
data["X34"] = data["X9"].apply(lambda x:1 if x == 2 else 0)

data["X35"] = data["X10"].apply(lambda x:1 if x == 2 else 0)
data["X36"] = data["X10"].apply(lambda x:1 if x == 2 else 0)

data["X37"] = data["X11"].apply(lambda x:1 if x == 2 else 0)
data["X38"] = data["X11"].apply(lambda x:1 if x == 2 else 0)

data["X6"] = data["X33"]
data["X7"] = data["X34"]
data["X8"] = data["X35"]
data["X9"] = data["X36"]
data["X10"] = data["X37"]
data["X11"] = data["X38"]


data = data.drop(['X33','X34','X35','X36','X37','X38'], axis = 1)


#Checking the unbalance size

data.Y.value_counts()
#77.88% class 0
#22.12% class 1

coli = data["X"+str(1)]
avgi = np.average(coli)
stdi = np.std(coli)
data["X1"] = (data["X1"] - avgi)/stdi
#cv_X["X1"] = (cv_X["X1"] - avgi)/stdi
#Test_X["X1"] = (Test_X["X1"] - avgi)/stdi

#Oversampling will only be applied to Training data
Train_Y = []
Train_Y = data.iloc[1:18001,1:2]
Train_X = data.iloc[1:18001,2:35]

cv_Y = data.iloc[18001:24001,1:2]
cv_X = data.iloc[18001:24001,2:35]
Test_Y = data.iloc[24001:30001,1:2]
Test_X = data.iloc[24001:30001,2:35]



#Resampling Training Data
sm = SMOTE(ratio = 'minority', k_neighbors = 5)
Train_X_res, Train_Y_res = sm.fit_sample(Train_X,Train_Y)

print("Y=1(%): ",str(np.sum(Train_Y_res)/Train_X_res.shape[0]))

#Changing names for old model to function properly
Train_X_org = Train_X
Train_Y_org = Train_Y

Train_X = Train_X_res

Train_Y = Train_Y_res.reshape((Train_X.shape[0],1))

cv_X = cv_X.values
Test_X = Test_X.values
cv_Y = cv_Y.values
Test_Y = Test_Y.values


#Just implement the model



    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:05:52 2017

@author: Rodrigo
"""

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
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

import tensorflow as tf
import math
from tensorflow.python.framework import ops
import time


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

# Polinomial Features
k = 33
for i in range(1,33):
    for j in range(i,33):
        data['X'+str(k)] = np.multiply(data['X'+str(i)],data['X'+str(j)])
        k = k+1
        
        
#Principal Component Analysis
data_X = data.iloc[0:30001,2:563]

pca = PCA(n_components=0.99, svd_solver='full',copy=False)
pca.fit(data_X)
Variance_ratio = pca.explained_variance_ratio_
Acc_Var_ratio = np.cumsum(Variance_ratio)

data_X_redu = pca.fit_transform(data_X)

# Normalize data

col_max = np.amax(data_X_redu,axis=0)

col_min = np.amin(data_X_redu,axis=0)

for i in range(data_X_redu.shape[1]):
    coli = data_X_redu[:,i]
    avgi = np.average(coli)
    stdi = np.std(coli)
    data_X_redu[:,i] = (data_X_redu[:,i] - avgi)/stdi
    
    
#Separating Training, CV and test set


Train_Y = data.iloc[1:18001,1:2]
Train_X = data_X_redu[1:18001,:]

cv_Y = data.iloc[18001:24001,1:2]
cv_X = data_X_redu[18001:24001,:]
Test_Y = data.iloc[24001:30001,1:2]
Test_X = data_X_redu[24001:30001,:]
cv_Y = cv_Y.values
Train_Y = Train_Y.values
Test_Y = Test_Y.values


#Smote



# Neural Nets

def create_placeholders(n,yj):
    X = tf.placeholder(tf.float32, name = 'X', shape = (None,n))
    Y = tf.placeholder(tf.float32, name = 'Y', shape = (None,yj))
    
    
    return X,Y


def initialize_parameters(layer_dims):
    """
    Arguments:
        layer_dims -- a Python array(list) containing the dimensions of each layer in the NN
        
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, ..... Wl,bl
                      Wl = matrix of shape (layer_dims[l],layer_dims[l-1])
                      b1 = bias of shape (layer_dims[l],1)
"""


    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = tf.get_variable("W"+ str(l), [layer_dims[l-1],layer_dims[l]], initializer = tf.contrib.layers.xavier_initializer())
        parameters['b'+str(l)] = tf.get_variable("b"+str(l),[1,layer_dims[l]],initializer = tf.zeros_initializer())
        
    return parameters

   

def random_mini_batches(X,Y,mini_batch_size=64):
    
    """Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (10, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]
    mini_batches = []
    
    #Shuffle X and Y
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    
    # partition 
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1)* mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1)* mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    #handling the end case (last batch that may not have same number os examples)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches : m,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches : m,:]
        ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

    return mini_batches

def linear_forward(A,W,b):
    Z = tf.add(tf.matmul(A,W),b)
    return Z

def relu_activation(Z):
     A  = tf.nn.relu(Z)
     return A
 
def L_model_forward(X,parameters,keep_probs):
    
    A = X
    L = len(parameters)//2
    
    # Relu activation function
    for l in range(1,L):
        A_prev = A
        A_prev = tf.nn.dropout(A_prev,keep_probs)
        A = relu_activation(linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)]))
        
    #last layer
    AL = linear_forward(A,parameters['W'+str(L)],parameters['b'+str(L)])
    
    return AL

def regularization(parameters,lamb=0.1):
    
    regularizer = 0
    L = len(parameters)//2
    
    for l in range(1,L+1):
        regularizer = regularizer + lamb * tf.add(tf.nn.l2_loss(parameters['W'+str(l)]),tf.nn.l2_loss(parameters['b'+str(l)]))
    
    return regularizer

def compute_cost(Z3,Y):
    """
Computes the cost
Arguments:
Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6,
number of examples)
Y -- "true" labels vector placeholder, same shape as Z3
Returns:
cost - Tensor of the cost function
"""
    # The function tf.nn.softmax_cross_entropy_with_logits require transpose
    logits = Z3
    labels = Y

    
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
    
    return cost

def model(Train_X,Train_Y,cv_X,cv_Y,Test_X,Test_Y,layer_dims,minibatch_size=32,alfa = 0.0001,lamb=0,keep_probs=1,num_epochs=1500,print_cost = True):
    """
    Train_X,Train_Y,cv_X,cv_Y,Test_X,Test_Y = Input your datasets
    layer_dims: a Python array(list) containing the dimensions of each layer in the NN
    minibatch_size: size of the minibatch. Select the number 2**X, with X=[5,10] for better performance
    alfa = Learning rate. Select a small enough for your model not to diverge, and big enough for fast optimization
    lamb = Regularization Lambda. Increase to avoid overfitting. Reduce to avoid bias
    num_epochs = Number of iterations of the model
    Print_cost = True if you want the costs to be printed as the model is learning
    """
    tic = time.time()
    ops.reset_default_graph()
    (m,n) = Train_X.shape # m = training examples n= number of features
    costs = []
    yj = Train_Y.shape[1]
    global_step = tf.Variable(0,trainable=False) #decaying alpha
    #keep_probs = tf.placeholder(tf.float32)
    
    X, Y = create_placeholders(n, yj)
    parameters = initialize_parameters(layer_dims)
    AL =L_model_forward(X, parameters,keep_probs)
    cost = compute_cost(AL, Y)
    regularizer = regularization(parameters,lamb)
    cost = tf.reduce_mean(regularizer)+cost
    #Decaying learning rate
    #decay_alfa = tf.train.exponential_decay(alfa,global_step,250,0.9,staircase=False)
    decay_alfa = tf.train.exponential_decay(alfa,global_step,250,0.9,staircase=False)
    #optimization
    optimizer = tf.train.AdamOptimizer(learning_rate = decay_alfa).minimize(cost,global_step=global_step)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            minibatches = random_mini_batches(Train_X,Train_Y,minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost = epoch_cost + minibatch_cost/num_minibatches
                
            if epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            costs.append(epoch_cost)
            
        
        #Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(alfa))
        plt.show()
        
        # Save parameters in the variable parameters
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        #Calculate predictions on Training Set
        #correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
        AL = tf.sigmoid(AL)
        prediction = tf.round(AL)
        
        correct_prediction = tf.equal(tf.cast(prediction, "int32"),tf.cast(Y, "int32"))
        accuracy = 100* tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy Training set: ", accuracy.eval({X: Train_X, Y: Train_Y}))
        print ("Test Accuracy CV set: ", accuracy.eval({X: cv_X, Y: cv_Y}))
        print ("Test Accuracy Test set: ", accuracy.eval({X: Test_X, Y: Test_Y}))
        
        pred_train = sess.run([prediction], feed_dict={X:Train_X})
        pred_cv = sess.run([prediction], feed_dict={X:cv_X})
        pred_test = sess.run([prediction], feed_dict={X:Test_X})
        
        AL_train = sess.run([AL], feed_dict={X:Train_X})
        AL_cv = sess.run([AL], feed_dict={X:cv_X})
        AL_test = sess.run([AL], feed_dict={X:Test_X})
        
        toc = time.time()
        print("Time: " + str(math.floor(toc-tic)))
        
        return parameters, pred_train, pred_cv, pred_test, AL_train, AL_cv, AL_test
    
    
    # Testing the model
    
layer_dims = [Train_X.shape[1],40,40,10,1]
parameters, pred_train, pred_cv, pred_test, AL_train, AL_cv, AL_test = model(Train_X,Train_Y,cv_X,cv_Y,Test_X,Test_Y,layer_dims,minibatch_size=256,alfa = 0.00025,lamb=0,keep_probs=0.85,num_epochs=1200,print_cost = True)

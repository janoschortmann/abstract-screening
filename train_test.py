#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:14:14 2023

@author: Isa
"""

import numpy as np
import pandas as pd
import os

import math
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
import sklearn.linear_model
import sklearn.utils
import sklearn.feature_selection
from sklearn.feature_selection import RFE
from sklearn.neighbors import LocalOutlierFactor
import joblib

directory = "/Users/isabel/"

outliers = 0 #1 if YES, 0 otherwise
n_neighbors = 20
contamination = 0.05

multi_collinearity = 0 #1 if YES, 0 otherwise
threshold_multi_collinearity = 0.95

oversampling = 0 #1 if YES, 0 otherwise
undersampling = 0
positives = 0.50

feature_selection = 0 #1 if YES, 0 otherwise
n_features_to_select = 0.60

n_splits = 10


os.chdir(directory)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#Read sample
data = np.load("training_testing_sets.npz",allow_pickle=True)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
vec_citations = data['vec_citations']

print("The training and test sets have been loaded.\n")

#####Remove Outliers
if outliers == 1:
    print("Outliers analysis is going to start.")
    print("There are originally {} data points in the training set.".format(len(X_train)))
    ##contamination is the proportion of outliers in the dataset
    ##n_neighbors is the number of nearest neighbors to consider when comparing 
    #the density of a datapoint using k-nearest neighbors
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    
    lof.fit(X_train)
    
    # Predict the labels of the training data
    # -1 indicates an outlier, 1 indicates an inlier
    y_lof_train = lof.fit_predict(X_train)
    outlier_indexes = np.where(y_lof_train == -1)
    
    print("There are {} outliers in the training set.".format(len(outlier_indexes)))
    
    X_train = np.delete(X_train, outlier_indexes, axis = 0)
    y_train = np.delete(y_train, outlier_indexes, axis = 0)
    
    print("Outliers have been dropped from the training set.\n")

#####REMOVE MULTICOLLINEARITY
if multi_collinearity == 1:

    print("Multicollinearity analysis is going to start.")
    print("There are originally {} features in the training set.".format(X_train.shape[1]))
    X_train = pd.DataFrame(X_train)

    #Correlation matrix
    corr_matrix = X_train.corr().abs()

    #Upper matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    #Obtain indexes in any column that has a  value bigger than the threshold
    multicollinearity_indexes = [column for column in upper.columns if any(upper[column] > threshold_multi_collinearity)]

    print("There are {} features with multicollinearity in the training dataset.".format(len(multicollinearity_indexes)))

    X_train = X_train.drop(X_train.columns[multicollinearity_indexes], axis=1)
    X_train = np.array(X_train)

    X_test = pd.DataFrame(X_test)
    X_test = X_test.drop(X_test.columns[multicollinearity_indexes], axis=1)
    X_test = np.array(X_test)
    
    vec_citations = pd.DataFrame(vec_citations)
    vec_citations = vec_citations.drop(vec_citations.columns[multicollinearity_indexes], axis=1)
    vec_citations = np.array(vec_citations)

    print("Features that have a high multicollinearity have been dropped from the training and testing sets.\n")

##Oversampling and Undersampling
y_train = np.array(y_train.reshape(len(y_train),), dtype = int)
y_test = np.array(y_test.reshape(len(y_test),), dtype = int)

X_positives = X_train[y_train==1]
y_positives = y_train[y_train==1]

X_negatives = X_train[y_train==0]
y_negatives = y_train[y_train==0]

print("The training dataset has a size of {}, with {} positives and {} negatives.".format(len(X_train),len(X_positives),len(X_negatives)))
print("The training dataset has a proportion of {} positives and {} of negatives.\n".format(round(len(y_positives)/len(y_train),2),round(len(y_negatives)/len(y_train),2)))

if positives == round(len(y_positives)/len(y_train),2):
    oversampling = 0
    undersampling = 0
    print("It is not possible to do undersampling/oversampling because the original proportion of positives is equal to the new intended proportion of positives.\n")
    
if oversampling==1 and undersampling==1:
    oversampling = 0
    undersampling = 0
    print("It is not possible to do undersampling and oversampling at the same time.\n")
    

if oversampling == 1:
    
    print("Oversampling is going to start.\n")
    A = np.array([[1, -1], [positives, -1]])
    b = np.array([len(X_negatives), 0])

    # Solve the system of equations
    x = np.linalg.solve(A, b)
    
    Total = math.ceil(x[0]) #TOTAL
    proportion_positives = math.ceil(x[1]) #new number of positives
    proportion_negatives = len(X_negatives) #new number of negatives
    
    print("The new intended total is {}, with {} positives and {} negatives.".format(Total, proportion_positives, proportion_negatives))
    print("The new intended proportion of positives is {} and of negatives is {}.\n".format(round(proportion_positives/Total,2), round(proportion_negatives/Total,2)))
    
    positives_toreplicate = proportion_positives - len(X_positives)
    
    print("{} replicas of positives are going to be generated...".format(positives_toreplicate))
    
    replicas = np.empty((0, X_positives.shape[1]))
    
    for i in range(0,positives_toreplicate):
        row_index = np.random.choice(X_positives.shape[0], size=1)[0]  # choose a random row index
        sampled_row = X_positives[row_index, :]
        sampled_row = sampled_row.reshape(1,len(sampled_row))
        replicas = np.append(replicas, sampled_row, axis=0)
    
    print("{} replicas of positives have been generated.".format(len(replicas)))
    
    X_positives = np.append(X_positives, replicas, axis = 0)
    y_positives = np.append(y_positives, np.ones((len(replicas),1)))
    
    print("{} replicas have been added and now there are {} positives.\n".format(len(replicas),len(X_positives)))
    
    X_train = np.append(X_positives,X_negatives,axis=0)
    y_train = np.append(y_positives,y_negatives,axis=0)
    
    print("Training set has now a size of {}.\n".format(len(X_train),len(y_train)))
    
if undersampling == 1:
    
    print("Undersampling is going to start.\n")
    negatives = 1-positives
    A = np.array([[positives, 0], [negatives, -1]])
    b = np.array([len(X_positives), 0])

    # Solve the system of equations
    x = np.linalg.solve(A, b)
    
    Total = math.ceil(x[0]) #TOTAL
    proportion_negatives = math.ceil(x[1]) #new number of negatives
    proportion_positives = len(X_positives) #new number of positives
    
    print("The new intended total is {}, with {} positives and {} negatives.".format(Total, proportion_positives, proportion_negatives))
    print("The new intended proportion of positives is {} and of negatives is {}.\n".format(round(proportion_positives/Total,2), round(proportion_negatives/Total,2)))
    
    negatives_todelete = len(X_negatives)-proportion_negatives
    
    print("{} negatives are going to be deleted...".format(negatives_todelete))
    
    for i in range(0,negatives_todelete):
        row_index = np.random.choice(X_negatives.shape[0], size=1)[0]  # choose a random row index
        X_negatives = np.delete(X_negatives,row_index,axis=0)
        y_negatives = np.delete(y_negatives,row_index,axis=0)
    
    print("{} negatives remained.".format(len(X_negatives)))
    
    X_train = np.append(X_positives,X_negatives,axis=0)
    y_train = np.append(y_positives,y_negatives,axis=0)
    
    print("Training set has now a size of {}.\n".format(len(X_train),len(y_train)))

#Train the Logistic regression with an L-2 penalty

log_reg = sklearn.linear_model.LogisticRegression(penalty = "l2", random_state = 0, max_iter = 1000).fit(X_train,y_train)

#Feature Selection
if feature_selection == 1:
    print("Feature selection is going to start.")
    print("There are originally {} features in the training dataset.".format(X_train.shape[1]))
    selector = RFE(log_reg, n_features_to_select=n_features_to_select)
    
    X_train = selector.fit_transform(X_train,y_train)
    X_test = selector.transform(X_test)
    vec_citations = selector.transform(vec_citations)
    
    log_reg = sklearn.linear_model.LogisticRegression(penalty = "l2", random_state = 0, max_iter = 1000).fit(X_train,y_train)

    print("After feature selection, now there are {} features in the training and testing sets.\n".format(X_train.shape[1]))
    
#Predict and obtain scores
y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)
y_prob_train = log_reg.predict_proba(X_train)
y_prob_test = log_reg.predict_proba(X_test)

score_train = round(accuracy_score(y_train,y_pred_train)*100, 2)
score_test = round(accuracy_score(y_test,y_pred_test)*100, 2)

recall_train = round(recall_score(y_train,y_pred_train)*100,2)
recall_test = round(recall_score(y_test,y_pred_test)*100, 2)

precision_train = round(precision_score(y_train,y_pred_train)*100, 2)
precision_test = round(precision_score(y_test,y_pred_test)*100, 2)

f1_train = round(f1_score(y_train,y_pred_train)*100, 2)
f1_test = round(f1_score(y_test,y_pred_test)*100, 2)

print("Probability prediction is going to start.\n")

print("LOGISTIC REGRESSION")

print("Confusion Matrix for Training Data")
print(pd.DataFrame(confusion_matrix(y_train, y_pred_train), 
             columns=['Predicted Negative', 'Predicted Positive'], 
             index=['Actual Negative', 'Actual Positive']))

print("The training has an accuracy of {}%, a recall of {}%, a precision of {}% and a f1-score of {}%.".format(score_train, recall_train, precision_train, f1_train))
print("\n")

print("Confusion Matrix for Test Data")
print(pd.DataFrame(confusion_matrix(y_test, y_pred_test), 
             columns=['Predicted Negative', 'Predicted Positive'], 
             index=['Actual Negative', 'Actual Positive']))

print("The testing has an accuracy of {}%, a recall of {}%, a precision of {}% and a f1-score of {}%.".format(score_test, recall_test, precision_test, f1_test))
print("\n")

if n_splits < len(X_positives) or n_splits < len(X_negatives):
    score_cv = cross_val_score(log_reg,X_train,y_train,cv=n_splits)
    score_cv = round(np.mean(score_cv)*100,2)
    
    recall_cv = cross_val_score(log_reg,X_train,y_train,scoring="recall",cv=n_splits)
    recall_cv = round(np.mean(recall_cv)*100,2)
    
    precision_cv = cross_val_score(log_reg,X_train,y_train,scoring="precision",cv=n_splits)
    precision_cv = round(np.mean(precision_cv)*100,2)
    
    f1_cv = cross_val_score(log_reg,X_train,y_train,scoring="f1",cv=n_splits)
    f1_cv = round(np.mean(f1_cv)*100,2)
    
    print("The accuracy of the 10-fold cross-validation is {}%, the recall is {}%, the precision is {}% and the f1-score is {}%.\n".format(score_cv, recall_cv, precision_cv, f1_cv))
else:
    print("Cross validation is not possible as the number of splits is greater than the number of elements in a class.\n")
    
save_lr = input("Would you like to save this logistic regression as your best model (y/n)?")
print(save_lr)

if save_lr == "y":
    np.savez('vec_citations.npz', vec_citations=vec_citations)
    print("Vectorized citations have been saved in npz file.\n")
    
    joblib.dump(log_reg, "logistic_regression.joblib")
    print("Logistic Regression has been saved in joblib file.\n")
else:
    print("Vectorized citations and Logistic Regression haven't been saved.")


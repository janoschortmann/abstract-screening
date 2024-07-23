#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:43:02 2024

@author: isabel
"""

import numpy as np
import pandas as pd
import os

import math
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
import sklearn.linear_model
from sklearn.tree import DecisionTreeClassifier
import sklearn.utils
import joblib

# Define the file path
file_path = "config4.txt"

# Initialize a dictionary to store the variables
variables = {}


with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        if line:

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')  

            variables[key] = value

# Access the variables
directory = variables.get("directory")
oversampling = int(variables.get("oversampling"))
undersampling = int(variables.get("undersampling"))
positives = float(variables.get("positives"))
n_splits = int(variables.get("n_splits"))
model_selection = variables.get("model_selection")

# Print the values
print("The configuration file has been read.")
print("oversampling:", oversampling)
print("undersampling:", undersampling)
print("positives:", positives)
print("n_splits:", n_splits)
print("model_selection:", model_selection)
print("directory:", directory)


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


print("\nThe training and test sets have been loaded.\n")


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

#Train the Logistic regression with an L-2 penalty or Decision Tree:

if model_selection == "lr":
    model = sklearn.linear_model.LogisticRegression(penalty = "l2", random_state = 0, max_iter = 1000).fit(X_train,y_train)

if model_selection == "dt":
    model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    
#Predict and obtain scores
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_prob_train = model.predict_proba(X_train)
y_prob_test = model.predict_proba(X_test)

score_train = round(accuracy_score(y_train,y_pred_train)*100, 2)
score_test = round(accuracy_score(y_test,y_pred_test)*100, 2)

recall_train = round(recall_score(y_train,y_pred_train)*100,2)
recall_test = round(recall_score(y_test,y_pred_test)*100, 2)

precision_train = round(precision_score(y_train,y_pred_train)*100, 2)
precision_test = round(precision_score(y_test,y_pred_test)*100, 2)

f1_train = round(f1_score(y_train,y_pred_train)*100, 2)
f1_test = round(f1_score(y_test,y_pred_test)*100, 2)

print("Probability prediction is going to start.\n")

if model_selection == "lr":
    print("LOGISTIC REGRESSION")
    
if model_selection == "dt":
    print("DECISION TREE")

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
    score_cv = cross_val_score(model,X_train,y_train,cv=n_splits)
    score_cv = round(np.mean(score_cv)*100,2)
    
    recall_cv = cross_val_score(model,X_train,y_train,scoring="recall",cv=n_splits)
    recall_cv = round(np.mean(recall_cv)*100,2)
    
    precision_cv = cross_val_score(model,X_train,y_train,scoring="precision",cv=n_splits)
    precision_cv = round(np.mean(precision_cv)*100,2)
    
    f1_cv = cross_val_score(model,X_train,y_train,scoring="f1",cv=n_splits)
    f1_cv = round(np.mean(f1_cv)*100,2)
    
    print("The accuracy of the 10-fold cross-validation is {}%, the recall is {}%, the precision is {}% and the f1-score is {}%.\n".format(score_cv, recall_cv, precision_cv, f1_cv))
else:
    print("Cross validation is not possible as the number of splits is greater than the number of elements in a class.\n")
    
if model_selection == "lr":
    save_model = input("Would you like to save this logistic regression as your best model (y/n)?")

if model_selection == "dt":
    save_model = input("Would you like to save this decision tree as your best model (y/n)?")

print(save_model)


if save_model == "y" and model_selection == "lr":
    joblib.dump(model, "model.joblib")
    print("Logistic Regression has been saved in joblib file.\n")

if save_model == "n" and model_selection == "lr":
    print("Logistic Regression hasn't been saved.")
    
if save_model == "y" and model_selection == "dt":
    joblib.dump(model, "model.joblib")
    print("Decision Tree has been saved in joblib file.\n")

if save_model == "n" and model_selection == "dt":
    print("Decision Tree hasn't been saved.")
    
if save_model != "n" and save_model != "y":
    print("Error: Answer not recognized.")
    

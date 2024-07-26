#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:04:47 2024

@author: isabel
"""

import numpy as np
import pandas as pd
import os
import math
import random
import sys

import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import sklearn.linear_model
from sklearn.tree import DecisionTreeClassifier
import sklearn.utils


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
test_size = float(variables.get("test_size"))
random_state = int(variables.get("random_state"))
oversampling = int(variables.get("oversampling"))
undersampling = int(variables.get("undersampling"))
positives = float(variables.get("positives"))
n_splits = int(variables.get("n_splits"))
model_selection = variables.get("model_selection")
step = float(variables.get("step"))
acceptance_sampling_n = int(variables.get("acceptance_sampling_n"))


# Print the values
print("The configuration file has been read.")
print("directory:", directory)
print("test_size:", test_size)
print("random_state:", random_state)
print("oversampling:", oversampling)
print("undersampling:", undersampling)
print("positives:", positives)
print("n_splits:", n_splits)
print("model_selection:", model_selection)
print("step:", step)
print("acceptance_sampling_n:", acceptance_sampling_n)



os.chdir(directory)

#VECTORIZATION-SPLIT SCRIPT

#Read sample and citations
pathfile_sample = directory + "sample.csv"
sample = pd.read_csv(pathfile_sample,header=0,encoding = 'unicode_escape')
full_texts_sample = sample["Title"] + " " + sample["Abstract"] + " " + sample["Journal"]
#Read response variable from sample
response_variable = np.array(sample["Label"]).reshape((len(sample),1))
full_texts_sample = np.array(full_texts_sample).reshape(len(full_texts_sample),1)
sample = np.array(sample)

pathfile_citations = directory + "citations_retrieved.csv"
citations = pd.read_csv(pathfile_citations,header=0)
full_texts = citations["Title"] + " " + citations["Abstract"] + " " + citations["Journal"]
full_texts = np.array(full_texts).reshape(len(full_texts),1)

pathfile_validation = directory + "validation.csv"
validation = pd.read_csv(pathfile_validation,header=0)
full_texts_validation = validation["Title"] + " " + validation["Abstract"] + " " + validation["Journal"]
full_texts_validation = np.array(full_texts_validation).reshape(len(full_texts_validation),1)

print("\nThe {} citations from the sample set have been loaded.\n".format(len(full_texts_sample)))
print("The {} citations from the validation set have been loaded.\n".format(len(full_texts_validation)))
print("The {} citations to classify have been loaded.\n".format(len(full_texts)))
print("The response variable from the {} citations of the sample set have been loaded.\n".format(len(response_variable)))

#Read N-grams
pathfile_ngrams = directory + "n_grams.csv"
n_grams = np.array(pd.read_csv(pathfile_ngrams,header=None))
n_grams = n_grams[:,0].reshape((len(n_grams),1))

print("{} N-grams have been loaded.\n".format(len(n_grams)))


#Inicializar vectorizacion
print("Vectorization is going to start.")

vec = np.zeros((len(full_texts_sample),len(n_grams)))

for i in range(len(full_texts_sample)):
    for j in range(len(n_grams)):
        if n_grams[j,0] in full_texts_sample[i,0]:
            vec[i,j] = full_texts_sample[i,0].count(n_grams[j,0])
            
vec_validation = np.zeros((len(full_texts_validation),len(n_grams)))

for i in range(len(full_texts_validation)):
    for j in range(len(n_grams)):
        if n_grams[j,0] in full_texts_validation[i,0]:
            vec_validation[i,j] = full_texts_validation[i,0].count(n_grams[j,0])
            
vec_citations = np.zeros((len(full_texts),len(n_grams)))

for i in range(len(full_texts)):
    for j in range(len(n_grams)):
        if n_grams[j,0] in full_texts[i,0]:
            vec_citations[i,j] = full_texts[i,0].count(n_grams[j,0])

print("Vectorization has finished.\n")

#Split into raining and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(vec,response_variable,test_size=test_size,random_state = random_state)
print("The vectorized sample set has been split into training and testing sets with sizes {} and {} respectively.\n".format(len(X_train),len(X_test)))

y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)

X_train_positives = X_train[y_train==1]
y_train_positives = y_train[y_train==1]

X_train_negatives = X_train[y_train==0]
y_train_negatives = y_train[y_train==0]

print("The training dataset has a size of {}, with {} positives and {} negatives.".format(len(X_train),len(X_train_positives),len(X_train_negatives)))
print("The training dataset has a proportion of {} positives and {} of negatives.\n".format(round(len(y_train_positives)/len(y_train),2),round(len(y_train_negatives)/len(y_train),2)))

X_test_positives = X_test[y_test==1]
y_test_positives = y_test[y_test==1]

X_test_negatives = X_test[y_test==0]
y_test_negatives = y_test[y_test==0]

print("The testing dataset has a size of {}, with {} positives and {} negatives.".format(len(X_test),len(X_test_positives),len(X_test_negatives)))
print("The testing dataset has a proportion of {} positives and {} of negatives.\n".format(round(len(y_test_positives)/len(y_test),2),round(len(y_test_negatives)/len(y_test),2)))



#TRAINING-TESTING SCRIPT
y_positives = y_train_positives
y_negatives = y_train_negatives
X_negatives = X_train_negatives
X_positives = X_train_positives


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


# Handle user response
if model_selection == "lr":
    save_model = input("Would you like to use this logistic regression (y/n)?")

if model_selection == "dt":
    save_model = input("Would you like to use this decision tree (y/n)?")

print(save_model)

if save_model == "n":
    print("Exiting the script as per the user's choice.")
    sys.exit(0)
elif save_model == "y":
    print("Continuing with the code as per the user's choice.")
    # Continue with the rest of your code here
else:
    print("Error: Answer not recognized.")
    sys.exit(1)
    

#IMPLEMENTATION SCRIPT
threshold = 1

#Rest of the corpus and validation set
probabilities = model.predict_proba(vec_citations)
probabilities = probabilities[:,1].reshape(len(probabilities),1)

probabilities_validation = model.predict_proba(vec_validation)
probabilities_validation = probabilities_validation[:,1].reshape(len(probabilities_validation),1)
print("The inclusion probability of the citations has been calculated.\n")

citations_proba = np.append(citations,probabilities, axis=1)
citations_proba = pd.DataFrame(citations_proba, columns = ['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI', 'Probability of Inclusion'])
sorted_citations_proba = citations_proba.sort_values(by='Probability of Inclusion', ascending=False)

validation_proba = np.append(validation,probabilities_validation, axis=1)
validation_proba = pd.DataFrame(validation_proba, columns = ['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI', 'Probability of Inclusion'])
sorted_validation_proba = validation_proba.sort_values(by='Probability of Inclusion', ascending=False)

sorted_citations_proba.to_csv("citations_retreived_probabilities.csv", header=True, index=False)
sorted_validation_proba.to_csv("validation_probabilities.csv", header=True, index=False)

print("Citations have been saved in decscending order according to their probability of inclusion in a csv file.")


citations_proba = np.array(citations_proba)
validation_proba = np.array(validation_proba)

citation_intervals = int(1/step)
threshold_intervals = int((1-(1-threshold))/step)

intervals = [np.array([]) for i in range(citation_intervals)]

citation_sample = [np.array([]) for i in range(threshold_intervals)]

extreme_1 = 0
extreme_2 = step

for i in range(0,citation_intervals):  
    interval = np.empty((1,6))
    for c in range(0,len(probabilities)):
        if (probabilities[c] > extreme_1 and probabilities[c] <= extreme_2):
            interval = np.append(interval, citations_proba[c].reshape(1,6), axis=0)
    
    interval = np.delete(interval, 0, axis=0)
    intervals[i] = interval
    print("The size of interval {} between {} and {} is {}.".format(i,round(extreme_1,2),round(extreme_2,2),len(intervals[i])))
    
    if i < threshold_intervals:
        if acceptance_sampling_n > len(interval):
            citation_sample[i] = interval
        else:
            index_sample = random.sample(range(0,len(interval)),acceptance_sampling_n)
            citation_sample[i] = interval[index_sample,:]
    
    extreme_1 = extreme_1 + step
    extreme_2 = extreme_2 + step
    
print("\n")  
intervals = [np.array([]) for i in range(citation_intervals)]

extreme_1 = 0
extreme_2 = step

for i in range(0,citation_intervals):  
    interval = np.empty((1,6))
    for c in range(0,len(probabilities_validation)):
        if (probabilities_validation[c] > extreme_1 and probabilities_validation[c] <= extreme_2):
            interval = np.append(interval, validation_proba[c].reshape(1,6), axis=0)
    
    interval = np.delete(interval, 0, axis=0)
    intervals[i] = interval
    print("The size of the validation set in the interval {} between {} and {} is {}.".format(i,round(extreme_1,2),round(extreme_2,2),len(intervals[i])))
    
    extreme_1 = extreme_1 + step
    extreme_2 = extreme_2 + step


print("\n")
print("Sampling for acceptance sampling plan is going to start.")

extreme_1 = 0
extreme_2 = step
sample = np.empty((1,6))
    
for i in range(0,threshold_intervals):
    print("The size of the sample set in the interval {} between {} and {} is {}.".format(i,round(extreme_1,2),round(extreme_2,2),len(citation_sample[i])))
    extreme_1 = extreme_1 + step
    extreme_2 = extreme_2 + step
    sample = np.append(sample, citation_sample[i], axis=0)

sample = np.delete(sample, 0, axis=0)
sample = pd.DataFrame(sample, columns = ['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI', 'Probability of Inclusion'])
sample.to_csv("acceptance_sampling_plan_probabilities.csv", header=True, index=False)

print("\n")
print("Sample has been saved in csv.")



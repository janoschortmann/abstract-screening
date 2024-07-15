#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 00:41:20 2023

@author: Isa
"""
import numpy as np
import pandas as pd
import os
import sklearn.model_selection

# Define the file path
file_path = "config3.txt"

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
test_size = float(variables.get("test_size"))
random_state = int(variables.get("random_state"))
directory = variables.get("directory")

# Print the values
print("The configuration file has been read.")
print("test_size:", test_size)
print("random_state:", random_state)
print("directory:", directory)



os.chdir(directory)
#Read sample and citations
pathfile_sample = directory + "sample_revised.csv"
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

print("\nThe {} citations from the sample have been loaded.\n".format(len(full_texts_sample)))
print("The {} citations to classify have been loaded.\n".format(len(full_texts)))
print("The response variable from the {} citations of the sample have been loaded.\n".format(len(response_variable)))

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
            
vec_citations = np.zeros((len(full_texts),len(n_grams)))

for i in range(len(full_texts)):
    for j in range(len(n_grams)):
        if n_grams[j,0] in full_texts[i,0]:
            vec_citations[i,j] = full_texts[i,0].count(n_grams[j,0])

print("Vectorization has finished.\n")

#Split into raining and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(vec,response_variable,test_size=test_size,random_state = random_state)
print("The vectorized data set has been split into training and testing sets with sizes {} and {} respectively.\n".format(len(X_train),len(X_test)))

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


np.savez('training_testing_sets.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,vec_citations=vec_citations)
print("Training and testing sets, as well as the vectorization of the rest of the citations have been saved in npz file.\n")



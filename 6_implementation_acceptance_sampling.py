#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:12:30 2023

@author: Isa
"""
import numpy as np
import pandas as pd
import os
import joblib
import random


# Define the file path
file_path = "config6.txt"

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
step = float(variables.get("step"))
acceptance_sampling_n = int(variables.get("acceptance_sampling_n"))
directory = variables.get("directory")

# Print the values
print("The configuration file has been read.")
print("step:", step)
print("acceptance_sampling_n:", acceptance_sampling_n)
print("directory:", directory)


os.chdir(directory)

threshold = 1
data = np.load("training_testing_sets.npz",allow_pickle=True)
vec_citations = data["vec_citations"]
vec_validation = data["vec_validation"]
print("\nVectorized citations have been loaded.\n")

citations = np.array(pd.read_csv("citations_retrieved.csv",header=0))
validation = np.array(pd.read_csv("validation.csv",header=0))
print("Citations have been loaded.\n")

log_reg = joblib.load("model.joblib")
print("Model has been loaded.\n")

#Rest of the corpus and validation set
probabilities = log_reg.predict_proba(vec_citations)
probabilities = probabilities[:,1].reshape(len(probabilities),1)

probabilities_validation = log_reg.predict_proba(vec_validation)
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
    



        



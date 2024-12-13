#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:26:48 2023

@author: Isa
"""

import numpy as np
import pandas as pd
import os
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('punkt')


def read_variables_from_file(file_path):

    variables = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"')

                variables[key] = value

    return variables

# Access the variables
variables = read_variables_from_file("config2.txt")
min_frequency = int(variables.get("min_frequency"))

variables = read_variables_from_file("config1.txt")
directory = variables.get("directory")

# Print the values
print("The configuration file has been read.")
print("directory:", directory)
print("min_frequency:", min_frequency)

os.chdir(directory)

#Read sample
pathfile_sample = directory + "sample.csv"
sample = pd.read_csv(pathfile_sample,header=0,encoding = 'unicode_escape')
full_texts_sample = sample["Title"] + " " + sample["Abstract"] + " " + sample["Journal"]
full_texts_sample = np.array(full_texts_sample).reshape(len(full_texts_sample),1)

print("\nSample has been loaded.\n")

print("UNI-grams process is going to start.")
print("Tokenization is going to start.")
#Tokenization
text = []

for i in range((len(full_texts_sample))):
    words = wordpunct_tokenize(full_texts_sample[i,0].lower())#
    text = np.append(text,words)#

text = text.reshape((len(text),1))
print("Tokenization has finished and there are originally {} words in the {} citations.".format(len(text),len(full_texts_sample)))

#Remove Stopwords, punctuation marks and numbers
print("Stopwords, punctuation marks and numbers are going to be removed.")

sw = stopwords.words('english')

for i in range(0,len(sw)):
    text = np.delete(text, text[:,0]==sw[i], axis=0)
    
punct = list(string.punctuation)

for i in range(0,len(punct)):
    text = np.delete(text, text[:,0]==punct[i], axis=0)

mask = np.array([s.isnumeric() for s in text.reshape(len(text),)], dtype = bool)
text = np.delete(text, np.where(mask)[0])
text = text.reshape(len(text),1)


print("After removing a list of {} stop words, {} punctuation marks and all numbers, {} words remained.".format(len(sw),len(punct),len(text))) 

#Stemming
print("Stemming process is going to start.")

ps = PorterStemmer()

for i in range(0,len(text)):
    text[i,0] = ps.stem(text[i,0])
    

print("Stemming process has finished.")

#Identify unique UNI-grams and count them
text = text.reshape(len(text),)
uni_grams = Counter(text)

uni_grams = pd.DataFrame(list(uni_grams.items()))
uni_grams = uni_grams.sort_values(by=1, ascending=False)

uni_grams = np.array(uni_grams)
uni_grams = uni_grams[uni_grams[:,1]>=min_frequency]

print("There are {} unique UNI-grams with a min frequency of {}.\n".format(len(uni_grams),min_frequency))

print("{} UNI-grams have been obtained.\n".format(len(uni_grams)))

print("BI- and TRI-grams process is going to start.")

full_texts_sample = [item for sublist in full_texts_sample for item in sublist]

# Find bi-grams
vectorizer_bi = CountVectorizer(ngram_range=(2, 2), stop_words=sw)
X_bi = vectorizer_bi.fit_transform(full_texts_sample)
bi_grams = vectorizer_bi.get_feature_names_out()
X_bi = X_bi.toarray()

# Find tri-grams
vectorizer_tri = CountVectorizer(ngram_range=(3, 3), stop_words=sw)
X_tri = vectorizer_tri.fit_transform(full_texts_sample)
tri_grams = vectorizer_tri.get_feature_names_out()
X_tri = X_tri.toarray()
    
print("{} unique BI-grams have been obtained without considering {} stop words.".format(len(bi_grams), len(sw)))
print("{} unique TRI-grams have been obtained without considering {} stop words.".format(len(tri_grams), len(sw)))

bi_gram_freq = np.sum(X_bi, axis=0)
mask = bi_gram_freq >= min_frequency
bi_grams = bi_grams[mask]
X_bi = X_bi[:, mask]

tri_gram_freq = np.sum(X_tri, axis=0)
mask = tri_gram_freq >= min_frequency
tri_grams = tri_grams[mask]
X_tri = X_tri[:, mask]

print("There are {} unique BI-grams and {} TRI-grams with a min frequency of {}.".format(len(bi_grams),len(tri_grams),min_frequency))

def contains_digits(s):
    return any(char.isdigit() for char in s)

bi_grams = bi_grams.reshape(len(bi_grams),1)
tri_grams = tri_grams.reshape(len(tri_grams),1)

mask = np.isin(bi_grams[:,0], punct)
mask = ~mask
bi_grams = bi_grams[mask]

bi_grams = np.array([s for s in bi_grams if not contains_digits(s[0])], dtype=object)

mask = np.isin(tri_grams[:,0], punct)
mask = ~mask
tri_grams = tri_grams[mask]

tri_grams = np.array([s for s in tri_grams if not contains_digits(s[0])], dtype=object)

print("After removing BI-grams with at least one out of {} punctuation marks or a number, {} BI-grams remained.".format(len(punct),len(bi_grams)))
print("After removing TRI-grams with at least one out of {} punctuation marks or a number, {} TRI-grams remained.\n".format(len(punct),len(tri_grams)))

print("{} BI-grams and {} TRI-grams have been obtained.\n".format(len(bi_grams),len(tri_grams)))

n_grams = np.concatenate((uni_grams[:,0].reshape(len(uni_grams),1), bi_grams, tri_grams), axis=0)
n_grams = n_grams.reshape((len(n_grams),))

print("There is a total of {} N-grams.\n".format(len(n_grams)))

#Only keep N-grams that have more than 3 characters
mask = np.array([len(s) > 4 for s in n_grams], dtype = bool)
n_grams = n_grams[mask]

print("{} N-grams with more than 3 characters have been identified and kept.\n".format(len(n_grams)))

n_grams = pd.DataFrame(n_grams)

#SAVE FILES
pathfile_ngrams = directory + "n_grams.csv"
n_grams.to_csv(pathfile_ngrams, header=False, index=False)

print("The N-grams and their frequency have been saved in a csv file.")


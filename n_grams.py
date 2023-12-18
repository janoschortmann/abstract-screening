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
nltk.download('stopwords')
nltk.download('punkt')

min_frequency = 2
directory = "/Users/isabel/"

os.chdir(directory)

#Read sample
pathfile_sample = directory + "sample_revised.csv"
sample = pd.read_csv(pathfile_sample,header=0,encoding = 'unicode_escape')
full_texts_sample = sample["Title"] + " " + sample["Abstract"] + " " + sample["Journal"]
full_texts_sample = np.array(full_texts_sample).reshape(len(full_texts_sample),1)

print("\n")
print("Sample has been loaded.\n")

print("Tokenization is going to start.")
#Tokenization
text = []

for i in range((len(full_texts_sample))):
    words = wordpunct_tokenize(full_texts_sample[i,0].lower())#
    text = np.append(text,words)#

text = text.reshape((len(text),1))
print("Tokenization has finished and there are originally {} words in the {} citations.\n".format(len(text),len(full_texts_sample)))

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

print("After removing a list of {} stop words, {} punctuation marks and all numbers, {} words remained.\n".format(len(sw),len(punct),len(text))) 

#Stemming
print("Stemming process is going to start.")

ps = PorterStemmer()

for i in range(0,len(text)):
    text[i,0] = ps.stem(text[i,0])
    
print("Stemming process has finished and {} UNI-grams have been obtained.\n".format(len(text)))

#Only keep uni-grams that have more than 3 characters
text = text.reshape((len(text),))
mask = np.array([len(s) < 4 for s in text], dtype = bool)
text = np.delete(text, np.where(mask)[0])

print("{} UNI-grams with more than 3 characters have been identified and kept.\n".format(len(text)))

#Identify unique UNI-grams and count them
text_sorted = Counter(text)

text_sorted = pd.DataFrame(list(text_sorted.items()))
text_sorted = text_sorted.sort_values(by=1, ascending=False)

text_sorted = np.array(text_sorted)
text_sorted = text_sorted[text_sorted[:,1]>=min_frequency]
text_sorted = pd.DataFrame(text_sorted)

print("There are {} unique UNI-grams with a min frequency of {}.\n".format(len(text_sorted),min_frequency))

#SAVE FILES
pathfile_unigrams = directory + "unigrams.csv"
text_sorted.to_csv(pathfile_unigrams, header=False, index=False)

print("The UNI-grams and their frequency have been saved in a csv file.")

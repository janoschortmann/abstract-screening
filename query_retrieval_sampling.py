#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:21:28 2023

@author: Isa
"""

import numpy as np
import pandas as pd
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import AbstractRetrieval
import random
import math
import os

query = 'TITLE-ABS-KEY ("healthcare logistics")'
limit_year = '2000'
sample_size = 0.2
directory = "/Users/Gustavo/"

os.chdir(directory)
#Query run
print("Query run is going to start.")
result = ScopusSearch(query)
result_size = result.get_results_size()
eid = np.array(result.get_eids())
print("{} citations met the query specifications.\n".format(result_size))

#Retrieve citations
print("Citation retrieving process is going to start.")
abstracts = []
titles = []
journals = []
dois = []
full_texts = []
missing = []
dates = []
errors = []
accepted = []

for i in range(len(eid)):
    try:
        ab = AbstractRetrieval(eid[i,])
        date = ab.coverDate
        
        if limit_year in date:
            break;
            
        abstract = ab.description
        if abstract is None:
            missing = np.append(missing,eid[i,])
        else:
            dates = np.append(dates,date)
            abstracts = np.append(abstracts,abstract)
            title = ab.title
            titles = np.append(titles,title)
            
            journal = ab.publicationName
            if journal is None:
                journal = "Unknown"
            journals = np.append(journals,journal)
            
            doi = ab.doi
            if doi is None:
                doi = "Unknown"
            dois = np.append(dois,doi)
            
            space = ' '
            full_text = title + space + abstract + space + journal
            full_texts = np.append(full_texts,full_text)
            accepted = np.append(accepted,eid[i,])
            
    except:
        errors = np.append(errors,eid[i,])
        pass
    
    if i % 100 == 0:
        print("{} citations have been called.".format(i))
        
print("Citation retrieving process has finished.\n")
print("{} citations were missing.\n".format(len(missing)))
print("{} citations retrieved an error.\n".format(len(errors)))
print("{} citations were retrieved.\n".format(len(titles)))

citations = np.stack((titles,abstracts,journals,dates,dois,accepted),axis=1)

#Sampling
print("Sampling is going to start.")

total = math.ceil(len(citations)*sample_size)
index_sample = random.sample(range(0,len(citations)),total)
citation_sample = citations[index_sample,:]
full_texts = full_texts.reshape(len(full_texts),1)
full_texts_sample = full_texts[index_sample,:]

print("{} citations were sampled, which represent {}% of the total corpus.".format(total,round((len(citation_sample)/len(citations))*100,2)))

citations = np.delete(citations,index_sample,axis=0)
full_texts = np.delete(full_texts,index_sample,axis=0)

print("Therefore, after removing the sample from the corpus, {} citations remained.\n".format(len(citations)))

#Saving files
np.savez('citations_retrieved_sample.npz', citations=citations, citation_sample=citation_sample, missing=missing, errors=errors, full_texts=full_texts, full_texts_sample=full_texts_sample)
print("Citations retrieved and sample have been saved in npz file.\n")


pathfile_citations = directory + "citations_retrieved.csv"
pathfile_sample = directory + "sample.csv"

labels = np.full((len(citation_sample),1),np.nan)

citation_sample = np.concatenate((citation_sample,labels),axis=1)

df_citations = pd.DataFrame(citations, columns=['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI', 'EID'])
df_citation_sample = pd.DataFrame(citation_sample, columns=['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI', 'EID', 'Label'])

df_citations.to_csv(pathfile_citations, header=True, index=False)
df_citation_sample.to_csv(pathfile_sample, header=True, index=False)

print("Citations retrieved and sample have been saved in csv file.\n")











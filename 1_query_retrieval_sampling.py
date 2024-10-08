#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:23:45 2024

@author: isabel
"""
import numpy as np
import pandas as pd
import os
import requests
from urllib.parse import quote
import random
import math


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
variables = read_variables_from_file("config1.txt")

api_key = variables.get("api_key")
token = variables.get("token")
query = variables.get("query")
date_range = variables.get("date_range")
sample_size = float(variables.get("sample_size"))
validation_size = float(variables.get("validation_size"))
total_records = int(variables.get("total_records"))
directory = variables.get("directory")

# Print the values
print("The configuration file has been read.")
print("directory:", directory)
print("api_key:", api_key)
print("token:", token)
print("query:", query)
print("date_range:", date_range)
print("sample_size:", sample_size)
print("validation_size:", validation_size)
print("total_records:", total_records)


headers = {
    'X-ELS-APIKey': api_key,
    'X-ELS-Insttoken': token #,
#    'Content-Type': 'application/x-www-form-urlencoded'
}

print("\nRetrieving process is going to start.")

os.chdir(directory)

doi = []
titles = []
journals = []
dates = []
abstracts = []
missing = []
total_processed_entries = 0

start = 0
count = 25  # Adjust based on your needs and API limits
view = 'COMPLETE'

while total_processed_entries < total_records:
    # Make the API request with the current start and count
    api_url = f'https://api.elsevier.com/content/search/scopus?apiKey={api_key}&query={quote(query)}&view={view}&start={start}&count={count}'
    print(f'Making API request to: {api_url}')  # Debugging: Print the request URL

    try:
        # Make the API request
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        # Process the response
        data = response.json()

        # Process the results
        search_results = data.get('search-results', {})
        entry = search_results.get('entry', [])
        if not entry:
            print("No more entries found. Exiting loop.")
            break

        for i in range(len(entry)):
            # Increment the counter for total processed entries
            total_processed_entries += 1

            # Process each entry
            current_entry = entry[i]
            if current_entry.get('dc:description', '') is None or current_entry.get('dc:description', '') == '':
                missing.append(current_entry.get('prism:doi', ''))
            else:
                abstracts.append(current_entry.get('dc:description', ''))
                doi_value = current_entry.get('prism:doi', '')
                doi.append("Unknown" if doi_value is None else doi_value)

                title_value = current_entry.get('dc:title', '')
                titles.append("Unknown" if title_value is None else title_value)

                journal_value = current_entry.get('prism:publicationName', '')
                journals.append("Unknown" if journal_value is None else journal_value)

                date_value = current_entry.get('prism:coverDate', '')
                dates.append("Unknown" if date_value is None else date_value)


        # Increment the start for the next batch
        start += count

        if total_processed_entries % 100 == 0:
            print(f"{total_processed_entries} abstracts have been called.")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        break

# Print summary information after processing all results
print(f"\nAbstract retrieving process has finished.")
print(f"{len(missing)} abstracts were missing.")
print(f"{len(titles)} papers met the year inclusion criteria.")

citations = np.stack((titles,abstracts,journals,dates,doi),axis=1)

#Sampling
print("\nSampling is going to start.")

total_sample = math.ceil(len(citations)*sample_size)

## the calculation of the validations et needs to be done on the original size of the corpus
total_validation = math.ceil(len(citations)*validation_size)
##

index_sample = random.sample(range(0,len(citations)),total_sample)
citation_sample = citations[index_sample,:]

print("{} citations were sampled for the sample set, which represent {}% of the total corpus.".format(total_sample,round((len(citation_sample)/len(citations))*100,2)))

citations = np.delete(citations,index_sample,axis=0)

print("Therefore, after removing the sample set from the corpus, {} citations remained.\n".format(len(citations)))


#Sampling (Validation)
index_validation = random.sample(range(0,len(citations)),total_validation)
citation_validation = citations[index_validation,:]

print("Then, {} citations were sampled for the validation set, which represent {}% of the total corpus.".format(total_validation,round((len(citation_validation)/len(citations))*100,2)))

citations = np.delete(citations,index_validation,axis=0)

print("Therefore, after removing the validation set from the corpus, {} citations remained.\n".format(len(citations)))


#Saving files

pathfile_citations = directory + "citations_retrieved.csv"
pathfile_sample = directory + "sample.csv"
pathfile_validation = directory + "validation.csv"

labels_sample = np.full((len(citation_sample),1),np.nan)

citation_sample = np.concatenate((citation_sample,labels_sample),axis=1)

df_citations = pd.DataFrame(citations, columns=['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI'])
df_citation_sample = pd.DataFrame(citation_sample, columns=['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI', 'Label'])
df_citation_validation = pd.DataFrame(citation_validation, columns=['Title', 'Abstract', 'Journal', 'Publishing Date', 'DOI'])

df_citations.to_csv(pathfile_citations, header=True, index=False)
df_citation_sample.to_csv(pathfile_sample, header=True, index=False)
df_citation_validation.to_csv(pathfile_validation, header=True, index=False)

print("Citations retrieved, sample set and validation set have been saved in csv files.\n")



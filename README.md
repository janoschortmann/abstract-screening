# AI-driven Comprehensive Abstract Screening (ACAS)

When performing a Systematic Literature Review (SLR), the Abstract Screening Process (ASP) can be a very consuming and laborious task, especially when researchers retrieve a significant number of citations after running queries in the selected databases. This can translate into many hours of work.  

This repository implements the Machine Learning-based methodology described in Serrato-Fonseca et al. (2024) and introduces the AI-based Comprehensive Abstract Screening (ACAS) tool that permits researchers to create their own ASP. This document is a detailed guideline to implementing said tool in any given SLR process.  

The tool works by viewing the choice of including a reference into the literature or not as a supervised machine learning (binary classification) problem. It allows the decision maker to leverage a small number of manually classified abstracts.


## How to use this tool

A detailed instruction manual is provided, see the file [instructions.pdf](instructions.pdf).

ACAS consists of four different Python scripts to be run sequentially. After running each of scripts 1 and 2, user intervention is required for adding manual inputs. 

1. Before running the scripts, adapt the parameters to your requirement by editing the configuration files [config1.txt](config1.txt) and [config2.txt](config2.txt).
2. Next, run [script 1](1_query_retrieval_sampling.py) (if using SCOPUS, to retrieve references)
3. Manually label the citations in [sample.csv](sample.csv).
4. Consider adding manual citations if known, to address the class imbalance issue.
5. Run [script 2](2_n_grams.py) to create the n-grams.
6. Revise the n-grams created and saved to [n_grams.csv](n_grams.csv) and consider adding further n-grams based on domain expertise.
7. Set up the remaining scripts by editing the configuration files [config3.txt](config3.txt) and [config4.txt](config4.txt).
8. Run [script 3](3_acceptance_sampling_plan.py)
9. Run [script 4](4_traintest prediction.py)


## Requirements 

The tool is written in Python 3.9 and requires the following Python packages: 
* numpy 2.0.0 
* pandas 2.2.2 
* requests 2.32.3
* nltk 3.8.1 
* sklearn 1.5.1 
* joblib 1.4.2 
* scipy 1.14.0
* matplotlib 3.9.1 

 
A full list of requirements can be found in the file _requirements.txt_. The user may wish to create either a conda environment or a virtualenv using this file. 

In addition to software environment described above, the user either needs a file of papers to be screened (the format is described in section 2.2 below) or a personal API key for the Scopus API that can be requested from [Elsevier](https://dev.elsevier.com/). Researchers affiliated to universities will typically be able to obtain access through their institution. 


## Scientific reference


Serrato-Fonseca, Julia Isabel; Anaya-Arenas, Ana María; Ortmann, Janosch; Ruiz, Angel. ACAS (2024). A comprehensive framework for automatic abstract screening in systematic literature reviews. _Cahiers du GERAD_ [G-2024-53](https://www.gerad.ca/en/papers/G-2024-53/).

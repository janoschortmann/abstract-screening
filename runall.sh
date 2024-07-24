#!/usr/bin/env bash


echo "press N to skip query retrieval"

read skip_retrieval

if [ "$skip_retrieval" = "N" ]
then
	echo "now running query retrieval"
	python 1_query_retrieval_sampling.py
else
	echo "skipping query retrieval"
fi

echo "Now running n-grams"
python 2_n_grams.py

echo "Now running vectorisation"
python 3_vectorization_split.py

echo "Now running model training"

python 4_train_test.py

echo "Now running acceptance sampling plan"

python 5_acceptance_sampling_plan.py

echo "Now implementing acceptance sampling plan"

python 6_implementation_acceptance_sampling.py



echo "done"
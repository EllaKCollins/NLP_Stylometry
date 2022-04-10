
# Generates csv files for the POS error counts

import pandas as pd
from collections import Counter
from convert_pos_counts import convert_pos_to_counts
from aligned_edits import retrieve_pos_from_dataframe 
from datasets import load_dataset

# Load dataset
train_pd = pd.read_csv('dataset/train.tsv', sep = '\t')
train_pd = train_pd[train_pd['modality'] != 'ht']
train_pd = train_pd.reset_index()
test_pd = pd.read_csv('dataset/test.tsv', sep = '\t')
test_pd = test_pd[test_pd['modality'] != 'ht']
test_pd = test_pd.reset_index()

# POS for train
print("Generating POS for train...")
new_train, ex = retrieve_pos_from_dataframe(train_pd)
x = convert_pos_to_counts(new_train)
x.to_csv('pos_df.csv')
ex.to_csv('ex.csv')
print("...done!")

# POS for test
print("Generating POS for test...")
new_train, ex = retrieve_pos_from_dataframe(test_pd)
x = convert_pos_to_counts(new_train)
x.to_csv('test_pos_df.csv')
ex.to_csv('test_ex.csv')
print("...done!")

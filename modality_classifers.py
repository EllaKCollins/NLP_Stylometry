
##########
# Imports
##########

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from text_preprocessing import modality_preprocessed_dataset, subject_preprocessed_dataset, test_modality_preprocessed_dataset

###############
# Load Dataset
###############

mdf, _ = modality_preprocessed_dataset()
sdf, _ = subject_preprocessed_dataset()

# Creating binary and sans-ht modality lists
mdf['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
pe_mdf = mdf[mdf['modality_2'] != 'ht']

###############################
# Logistic Regression Training
###############################

# All 3 modalities
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All 3 modalities")
print(classification_report(y, pred))
tri_lm = log_reg.fit(X, y)

# Binary modalities
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)
scores = cross_val_score(log_reg, X, y, cv = 5)

print("==================")
print("Binary Modalities")
print(classification_report(y, pred))
bi_lm = log_reg.fit(X, y)

# Only PE modalities
X = pe_mdf.drop(['modality', 'modality_2'], axis = 1)
y = pe_mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Only machine-translation Modalities")
print(classification_report(y, pred))
pe_lm = log_reg.fit(X, y)

###############################
# Testing the final models
###############################

test_df, _ = test_modality_preprocessed_dataset()
test_df['modality_2'] = ["pe" if i != "ht" else "ht" for i in test_df['modality']]
pe_test_df = test_df[test_df['modality_2'] != 'ht']

X = test_df.drop(['modality', 'modality_2'], axis = 1)
y = test_df['modality']
pred = log_reg.predict(X)

print(classification_report(y, pred))

###########################
# Use informative features
###########################

#from informative_features import plot_coefficients
#
#plot_coefficients(log_reg, X.columns, top_features = 20)


# Feature analysis for subject_id

##########
# Imports
##########

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from sklearn.feature_selection import f_classif, chi2
from text_preprocessing import modality_preprocessed_dataset, subject_preprocessed_dataset
import warnings
import sys
warnings.filterwarnings('ignore')

###############
# Load Dataset
###############

sdf_t, sdf, sdf_2 = subject_preprocessed_dataset()

# POS preprocessing
ex = pd.read_csv('ex.csv')
pesdf = sdf_t.drop(ex['index'])
pesdf = pesdf[pesdf['modality'] != 'ht'].reset_index().drop(columns = ['index'])

# POS error counts dataframe
sdf_e = pd.read_csv('pos_df.csv').drop(columns = ['subject_id', 'modality', 'Unnamed: 0', 'item_id'])
pesdfe = pesdf.join(sdf_e)

####################
# ANOVA with F-Test
####################

X = pesdfe.drop(columns = ['subject_id', 'modality'])
y = pesdf['subject_id']

f, p, = f_classif(X, y)

x = pd.DataFrame()
x["names"] = X.columns
x["f_value"] = f
x["p_value"] = p
x = x.sort_values(by = ["f_value"], ascending = False)
j = x[x["p_value"] > 0.05]
x = x[x["p_value"] < 0.05]
sig_feats = x["names"]
non_sig_feats = j["names"]

print(x)

######################
# Logistic Regression
######################

# Only Behavioral
X = sdf.drop(columns = ['subject_id', 'modality'])
y = sdf['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Without words")
print(classification_report(y, pred))

# Only words
X = sdf_2
y = sdf['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Only words")
print(classification_report(y, pred))

# Only POS Errors
X = sdf_e
y = pesdf['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Only POS Errors")
print(classification_report(y, pred))

# Altogether
X = pesdfe.drop(columns = ['subject_id', 'modality'])
y = pesdf['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Behavioral, TFIDF, and POS errors")
print(classification_report(y, pred))

# Only Significant Features
X = pesdfe[sig_feats]
y = pesdf['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Only significant features")
print(classification_report(y, pred))

## Only non-significant features
#X = sdf_t[non_sig_feats]
#y = sdf_t['subject_id']
#
#log_reg = LogisticRegression(max_iter = 200)
#pred = cross_val_predict(log_reg, X, y, cv = 5)
#
#print("==================")
#print("Non-significant features")
#print(classification_report(y, pred))
#
## Removed k_nav and num_annotations
#spec_feats = ['k_nav', 'num_annotations']
#X = sdf_t[spec_feats]
#y = sdf_t['subject_id']
#
#log_reg = LogisticRegression(max_iter = 200)
#pred = cross_val_predict(log_reg, X, y, cv = 5)
#
#print("==================")
#print("Only k_nav and num_annotations")
#print(classification_report(y, pred))

###########################
# Use informative features
###########################

#from informative_features import plot_coefficients
#
#plot_coefficients(log_reg, X.columns, top_features = 20)

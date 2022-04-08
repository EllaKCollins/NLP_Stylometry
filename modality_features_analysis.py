
# Feature analysis for modality

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

###############
# Load Dataset
###############

mdf_t, mdf, mdf_2 = modality_preprocessed_dataset()

# Creating binary and sans-ht modality lists
mdf['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
mdf_t['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
mdf_2['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
pe_mdf = mdf[mdf['modality_2'] != 'ht']

####################
# ANOVA with F-Test
####################

# Text classification
X = mdf_t.drop(['modality', 'modality_2'], axis = 1)
y = mdf_t['modality']

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

# 3 mod Without words
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All 3 modalities without words")
print(classification_report(y, pred))

# 3 mod only words
X = mdf_2.drop('modality_2', axis = 1)
y = mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All 3 modalities only words")
print(classification_report(y, pred))

# 3 mod with words
X = mdf_t.drop(['modality', 'modality_2'], axis = 1)
y = mdf_t['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All 3 modalities with words")
print(classification_report(y, pred))

# Removed non-significant features, 3 mods
X = mdf_t[sig_feats]
y = mdf_t['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("3 modalities, significant features")
print(classification_report(y, pred))

# Binary modality without words
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modalities without words")
print(classification_report(y, pred))

# Binary modality only words
X = mdf_2.drop('modality_2', axis = 1)
y = mdf_2['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modality only words")
print(classification_report(y, pred))

# Binary modality with words
X = mdf_t.drop(['modality', 'modality_2'], axis = 1)
y = mdf_t['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modalities with words")
print(classification_report(y, pred))

## With words
#X = mdf_t.drop('modality', axis = 1)
#y = mdf_t['modality']
#
#log_reg = LogisticRegression(max_iter = 200)
#scores = cross_val_score(log_reg, X, y, cv = 5)
#
#print("==================")
#print("With words")
#print(f"Cross-validation accuracy: {np.mean(scores):.4f}")
#

## Only non-significant features
#X = mdf_t[non_sig_feats]
#y = mdf_t['modality']
#
#log_reg = LogisticRegression(max_iter = 200)
#scores = cross_val_score(log_reg, X, y, cv = 5)
#
#print("==================")
#print("Only non-significant features")
#print(f"Cross-validation accuracy: {np.mean(scores):.4f}")
#
## Removed k_nav and num_annotations
#spec_feats = ['k_nav', 'num_annotations']
#X = mdf_t[sig_feats]
#X = X.drop(['k_nav', 'num_annotations'], axis = 1)
#y = mdf_t['modality']
#
#log_reg = LogisticRegression(max_iter = 200)
#scores = cross_val_score(log_reg, X, y, cv = 5)
#
#print("==================")
#print(f"Without Specific Features: {spec_feats}")
#print(f"Cross-validation accuracy: {np.mean(scores):.4f}")
#
## Removed k_nav and num_annotations
#spec_feats = ['k_nav', 'num_annotations']
#X = mdf_t[spec_feats]
#y = mdf_t['modality']
#
#log_reg = LogisticRegression(max_iter = 200)
#scores = cross_val_score(log_reg, X, y, cv = 5)
#
#print("==================")
#print(f"Specific Features: {spec_feats}")
#print(f"Cross-validation accuracy: {np.mean(scores):.4f}")

#for feat in sig_feats:
#    X = mdf_t.drop(['modality', feat], axis = 1)
#    y = mdf_t['modality']
#
#    log_reg = LogisticRegression(max_iter = 200)
#    scores = cross_val_score(log_reg, X, y, cv = 5)
#
#    print("==================")
#    print(f"Without {feat}")
#    print(f"Cross-validation accuracy: {np.mean(scores):.4f}")


###########################
# Use informative features
###########################

#from informative_features import plot_coefficients
#
#plot_coefficients(log_reg, X.columns, top_features = 20)

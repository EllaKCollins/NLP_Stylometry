
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

###############
# Load Dataset
###############

mdf, _, _ = modality_preprocessed_dataset()
sdf_t, sdf, sdf_2 = subject_preprocessed_dataset()

# Creating binary and sans-ht modality lists
mdf['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
pe_mdf = mdf[mdf['modality_2'] != 'ht']

####################
# ANOVA with F-Test
####################

# Text classification
X = sdf_t.drop('subject_id', axis = 1)
y = sdf_t['subject_id']

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

# Without words
X = sdf.drop('subject_id', axis = 1)
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

# With words
X = sdf_t.drop('subject_id', axis = 1)
y = sdf_t['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("With Words")
print(classification_report(y, pred))

# Removed non-significant features
X = sdf_t[sig_feats]
y = sdf_t['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Only significant features")
print(classification_report(y, pred))

# Only non-significant features
X = sdf_t[non_sig_feats]
y = sdf_t['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Non-significant features")
print(classification_report(y, pred))

# Removed k_nav and num_annotations
spec_feats = ['k_nav', 'num_annotations']
X = sdf_t[spec_feats]
y = sdf_t['subject_id']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Only k_nav and num_annotations")
print(classification_report(y, pred))

#for feat in sig_feats:
#    X = sdf_t.drop(['subject_id', feat], axis = 1)
#    y = sdf_t['subject_id']
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

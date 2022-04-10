
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
from text_preprocessing import modality_preprocessed_dataset, subject_preprocessed_dataset, test_modality_preprocessed_dataset
import warnings
import sys
warnings.filterwarnings('ignore')

###############
# Load Dataset
###############

mdf_t, mdf, mdf_2 = modality_preprocessed_dataset()
tmdf_t, tmdf, tmdf_2 = test_modality_preprocessed_dataset()
mdf_e = pd.read_csv('pos_df.csv').drop(columns = ['subject_id', 'modality', 'Unnamed: 0', 'item_id'])
tmdf_e = pd.read_csv('test_pos_df.csv').drop(columns = ['subject_id', 'modality', 'Unnamed: 0', 'item_id'])

# Binary modalities
mdf['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
mdf_t['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
mdf_2['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
tmdf_t['modality_2'] = ["pe" if i != "ht" else "ht" for i in tmdf['modality']]

# PE-only modalities and POS Errors
ex = pd.read_csv('ex.csv')
pe_mdf = mdf.drop(ex['index'])
pe_mdf = pe_mdf[pe_mdf['modality_2'] != 'ht'].reset_index().drop(columns = ['index'])
pe_mdf_2 = mdf_2.drop(ex['index'])
pe_mdf_2 = pe_mdf_2[pe_mdf_2['modality_2'] != 'ht'].reset_index().drop(columns = ['index'])
pe_mdf_t = mdf_t.drop(ex['index'])
pe_mdf_t = pe_mdf_t[pe_mdf_t['modality_2'] != 'ht'].reset_index().drop(columns = ['index'])
pe_mdf_e = pe_mdf_t.join(mdf_e)
test_ex = pd.read_csv('test_ex.csv')
test_pe_mdf_t = tmdf_t.drop(test_ex['index'])
test_pe_mdf_t = test_pe_mdf_t[test_pe_mdf_t['modality_2'] != 'ht'].reset_index().drop(columns = ['index'])
test_pe_mdf_e = test_pe_mdf_t.join(tmdf_e)

####################
# ANOVA with F-Test
####################

# All modalities
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
multi_sig_feats = x["names"]
non_sig_feats = j["names"]

print("==================")
print("All modalities ANOVA; behavioral and TF-IDF features")
print(x)

# Binary modalities
X = mdf_t.drop(['modality', 'modality_2'], axis = 1)
y = mdf_t['modality_2']

f, p, = f_classif(X, y)

x = pd.DataFrame()
x["names"] = X.columns
x["f_value"] = f
x["p_value"] = p
x = x.sort_values(by = ["f_value"], ascending = False)
j = x[x["p_value"] > 0.05]
x = x[x["p_value"] < 0.05]
bin_sig_feats = x["names"]
non_sig_feats = j["names"]

print("==================")
print("Binary modalities ANOVA; behavioral and TF-IDF features")
print(x)

# PE Modalities
X = pe_mdf_e.drop(['modality', 'modality_2'], axis = 1)
y = pe_mdf_e['modality']

f, p, = f_classif(X, y)

x = pd.DataFrame()
x["names"] = X.columns
x["f_value"] = f
x["p_value"] = p
x = x.sort_values(by = ["f_value"], ascending = False)
j = x[x["p_value"] > 0.05]
x = x[x["p_value"] < 0.05]
pe_sig_feats = x["names"]
non_sig_feats = j["names"]

print("==================")
print("PE modalities ANOVA; Behavioral, TF-IDF, and POS Error features")
print(x)

######################
# Logistic Regression
######################

# All modalities, only behavioral
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All modalities, only behavioral")
print(classification_report(y, pred))

# All modalities, only TF-IDF
X = mdf_2.drop('modality_2', axis = 1)
y = mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All modalities, only TF-IDF")
print(classification_report(y, pred))

# All modalities, behavioral and TF-IDF
X = mdf_t.drop(['modality', 'modality_2'], axis = 1)
y = mdf_t['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All modalities, behavioral and TF-IDF")
print(classification_report(y, pred))

# All modalities, only significant
X = mdf_t[multi_sig_feats]
y = mdf_t['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("All modalities, only significant features")
print(classification_report(y, pred))

# Predict test data, all modalities with significant model
log_reg = LogisticRegression(max_iter = 200)
log_reg.fit(X, y)

test_X = tmdf_t[multi_sig_feats]
test_y = tmdf_t['modality']
pred = log_reg.predict(test_X)

print("==================")
print("Test score: all modalities, significant model")
print(classification_report(test_y, pred))

print("==================")

# Binary modality, only behavioral 
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modalities, only behavioral")
print(classification_report(y, pred))

# Binary modality, only words
X = mdf_2.drop('modality_2', axis = 1)
y = mdf_2['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modality, only words")
print(classification_report(y, pred))

# Binary modality, behavioral and TF-IDF
X = mdf_t.drop(['modality', 'modality_2'], axis = 1)
y = mdf_t['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modalities, behavioral and TF-IDF")
print(classification_report(y, pred))

# Binary modality, only significant
X = mdf_t[bin_sig_feats]
y = mdf_t['modality_2']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("Binary modalities, only significant features")
print(classification_report(y, pred))

# Predict test data, binary modality with significant model
log_reg = LogisticRegression(max_iter = 200)
log_reg.fit(X, y)

test_X = tmdf_t[bin_sig_feats]
test_y = tmdf_t['modality_2']
pred = log_reg.predict(test_X)

print("==================")
print("Test score: binary modalities, significant model")
print(classification_report(test_y, pred))

print("==================")

# PE-only; Only behaviour
X = pe_mdf.drop(['modality', 'modality_2'], axis = 1)
y = pe_mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("PE-only; only behavioral")
print(classification_report(y, pred))

# PE-only; Only TF-IDF
X = pe_mdf_2.drop(['modality_2'], axis = 1)
y = pe_mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("PE-only; only TF-IDF")
print(classification_report(y, pred))

# PE-only; Only POS
X = mdf_e
y = pe_mdf['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("PE-only; only POS")
print(classification_report(y, pred))

# PE-only; behavioral, TF-IDF, and POS Error counts
X = pe_mdf_e.drop(['modality', 'modality_2'], axis = 1)
y = pe_mdf_e['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("PE-only; behavioral, TF-IDF, and POS Error counts")
print(classification_report(y, pred))

# PE-only; Only significant
X = pe_mdf_e[pe_sig_feats]
y = pe_mdf_e['modality']

log_reg = LogisticRegression(max_iter = 200)
pred = cross_val_predict(log_reg, X, y, cv = 5)

print("==================")
print("PE-only; Only significant features")
print(classification_report(y, pred))

# Predict test data, PE-only modality with significant model
log_reg = LogisticRegression(max_iter = 200)
log_reg.fit(X, y)

test_X = test_pe_mdf_e[pe_sig_feats]
test_y = test_pe_mdf_e['modality']
pred = log_reg.predict(test_X)

print("==================")
print("Test score: PE-only modalities, significant model")
print(classification_report(test_y, pred))

# Predict test data using only significant model

# With words
#X = mdf_t.drop('modality', axis = 1)
#y = mdf_t['modality']
#
#log_reg = LogisticRegression(max_iter = 200)
#scores = cross_val_score(log_reg, X, y, cv = 5)
#
#print("==================")
#print("With words")
#print(f"Cross-validation accuracy: {np.mean(scores):.4f}")

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

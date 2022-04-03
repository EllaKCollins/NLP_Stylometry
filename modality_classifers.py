
##########
# Imports
##########

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from text_preprocessing import modality_preprocessed_dataset, subject_preprocessed_dataset

###############
# Load Dataset
###############

mdf = modality_preprocessed_dataset()
sdf = subject_preprocessed_dataset()

# Creating binary and sans-ht modality lists
mdf['modality_2'] = ["pe" if i != "ht" else "ht" for i in mdf['modality']]
pe_mdf = mdf[mdf['modality_2'] != 'ht']

######################
# Logistic Regression
######################

# All 3 modalities
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("All 3 modalities")
print(classification_report(y_test, pred))

# Binary modalities
X = mdf.drop(['modality', 'modality_2'], axis = 1)
y = mdf['modality_2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("Binarized modalities")
print(classification_report(y_test, pred))

# Only PE modalities
X = pe_mdf.drop(['modality', 'modality_2'], axis = 1)
y = pe_mdf['modality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("PE modalities")
print(classification_report(y_test, pred))

# Subject classification
X = sdf.drop('subject_id', axis = 1)
y = sdf['subject_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("Subject Classification")
print(classification_report(y_test, pred))

###########################
# Use informative features
###########################

#from informative_features import plot_coefficients
#
#plot_coefficients(log_reg, X.columns, top_features = 20)

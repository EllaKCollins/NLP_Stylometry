
##########
# Imports
##########

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from text_preprocessing import modality_preprocessed_dataset

###############
# Load Dataset
###############

df = modality_preprocessed_dataset()

# Creating binary and sans-ht modality lists
df['modality_2'] = ["pe" if i != "ht" else "ht" for i in df['modality']]
pe_df = df[df['modality_2'] != 'ht']

######################
# Logistic Regression
######################

# All 3 modalities
X = df.drop(['modality', 'modality_2'], axis = 1)
y = df['modality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("Alll 3 modalities")
print(classification_report(y_test, pred))

# Binary modalities
X = df.drop(['modality', 'modality_2'], axis = 1)
y = df['modality_2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("Binarized modalities")
print(classification_report(y_test, pred))

# Only PE modalities
X = pe_df.drop(['modality', 'modality_2'], axis = 1)
y = pe_df['modality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("PE modalities")
print(classification_report(y_test, pred))

###########################
# Use informative features
###########################

#from informative_features import plot_coefficients
#
#plot_coefficients(log_reg, X.columns, top_features = 20)

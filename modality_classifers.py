
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

# Change modality to binary; pe1 and pe2 into pe 
df['modality'] = ["pe" if i != "ht" else "ht" for i in df['modality']]

######################
# Logistic Regression
######################

X = df.drop('modality', axis = 1)
y = df['modality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print("Logistic Regression Classifier for Modality")
print(classification_report(y_test, pred))

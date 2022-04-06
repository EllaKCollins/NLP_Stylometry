
##########
# Imports
##########

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import string, re, joblib
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from text_preprocessing import preprocess_translations

######################################
# Load and create TF-IDF vectorizier
######################################

df = pd.read_csv('dataset/train.tsv', sep = '\t')

text = preprocess_translations(df["tgt_text"])

tf = TfidfVectorizer()
tf.fit(text)

joblib.dump(tf, "tfidf_italian.joblib")

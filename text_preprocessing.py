
##########
# Imports
##########

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import string, re
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

#######################
# Preprocess Functions 
#######################

def preprocess_translations(raw_text):

    # Remove punctuation
    pf_text = [[i for i in text if i not in string.punctuation] for text in raw_text]
    pf_text = ["".join(i) for i in pf_text]

    # Make text to lowercase
    lower_text = [i.lower() for i in pf_text]

    # Remove Italian stopwords and numbers
    it_stopwords = stopwords.words("italian")
    text = [i.split(" ") for i in lower_text]
    text = [[i for i in sentence if i not in it_stopwords] for sentence in text]
    text = [[j for j in [re.sub(r'^\d*$', '', i) for i in sentence] if j != ''] for sentence in text]
    text = [" ".join(i) for i in text]

    return text

def it_tfidf(text):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    return vectorizer

######################
# Feature Engineering
######################

def modality_preprocessed_dataset():

    # Grab dataframe
    df = pd.read_csv('dataset/train.tsv', sep = '\t')

    # Remove features useless for modality
    df = df.drop(['item_id', 'subject_id', 'n_insert', 'n_delete', 'n_substitute', 'n_shift', 'bleu', 'chrf', 'ter', 'aligned_edit', 'mt_text'], axis = 1)

    # mt_df = df[df['modality'] != 'ht']

    # Preprocess and vectorize text
    text = preprocess_translations(df['tgt_text'])
    vec = it_tfidf(text)
    text = vec.transform(text)

    # Transform TFIDF sparse matrix into pandas
    df_2 = pd.DataFrame(text.toarray(), columns = vec.get_feature_names_out())

    # Removing unnecessary features after vectorization
    df = df.drop(['src_text', 'tgt_text'], axis = 1)
    df_2 = df_2.drop("17", axis = 1)

    # Changing features into floats 
    for column in df.columns:
        if column != 'modality':
            df[column] = [float(i) for i in df[column]]

    # Scaling features
    scaler = StandardScaler()
    scaling_features = [i for i in df.columns if i != "modality"]
    df[scaling_features] = scaler.fit_transform(df[scaling_features])
    
    # Return combined dataframe
    return df.join(df_2) 

def subject_preprocessed_dataset():

    # Grab dataframe
    df = pd.read_csv('dataset/train.tsv', sep = '\t')

    # Remove features useless for modality
    df = df.drop(['item_id', 'modality', 'n_insert', 'n_delete', 'n_substitute', 'n_shift', 'bleu', 'chrf', 'ter', 'aligned_edit', 'mt_text'], axis = 1)

    # mt_df = df[df['modality'] != 'ht']

    # Preprocess and vectorize text
    text = preprocess_translations(df['tgt_text'])
    vec = it_tfidf(text)
    text = vec.transform(text)

    # Transform TFIDF sparse matrix into pandas
    df_2 = pd.DataFrame(text.toarray(), columns = vec.get_feature_names_out())

    # Removing unnecessary features after vectorization
    df = df.drop(['src_text', 'tgt_text'], axis = 1)
    df_2 = df_2.drop("17", axis = 1)

    # Changing features into floats 
    for column in df.columns:
        if column != 'subject_id':
            df[column] = [float(i) for i in df[column]]

    # Scaling features
    scaler = StandardScaler()
    scaling_features = [i for i in df.columns if i != "subject_id"]
    df[scaling_features] = scaler.fit_transform(df[scaling_features])
    
    # Return combined dataframe
    return df.join(df_2) 

################
# Main Function
################

if __name__ == "__main__":
    pass
       

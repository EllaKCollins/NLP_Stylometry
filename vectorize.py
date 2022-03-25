
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



dataset = load_dataset("GroNLP/ik-nlp-22_pestyle", "full", data_dir="NLP_data")
train_pd = dataset['train'].to_pandas()

corpus = train_pd["tgt_text"]

vectorizer = TfidfVectorizer()
vect = vectorizer.fit(corpus)
transf = vectorizer.transform(corpus)

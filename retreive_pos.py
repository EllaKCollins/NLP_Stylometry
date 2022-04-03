from datasets import load_dataset
import pandas as pd
import spacy
import re

pos_tagger = spacy.load("it_core_news_sm")

dataset = load_dataset("GroNLP/ik-nlp-22_pestyle", "full", data_dir="NLP_data")
train_pd = dataset['train'].to_pandas()
aligned_edits = train_pd["aligned_edit"]


edit = aligned_edits[1]
s_edit = edit.split("\\n")
for x in s_edit:
    print(x)
print()
ref = s_edit[0].replace("REF:", "").replace("HYP:", "").replace("EVAL:", "")
hyp = s_edit[1].replace("REF:", "").replace("HYP:", "").replace("EVAL:", "")

# all insertions / deletions are now *
ref = re.sub("\*+", "*", ref)
hyp = re.sub("\*+", "*", hyp)

# remove all punctuation other than ' and *
ref = re.sub("[^\w\d'*\s]+",'', " ".join(re.split("\s+", ref, flags=re.UNICODE)).strip())
hyp = re.sub("[^\w\d'*\s]+",'', " ".join(re.split("\s+", hyp, flags=re.UNICODE)).strip())

ref_arr = ref.split()
hyp_arr = hyp.split()
new_evals = []
for index, word in enumerate(ref_arr): # this gives us the edits for each word where N is nothing
    if word == hyp_arr[index]:
        new_evals.append("N")
    elif "*" in word: 
        new_evals.append("I")
    elif "*" in hyp_arr[index]:
        new_evals.append("D")
    else:
        new_evals.append("S")
print(new_evals)
# all the things are the same length now

# gotta figure out how to use the pos_tagger() now
# for pos 
# replace ' with space ? 
# remove * that are left ? without that messing up everything ??

# hyp = pos_tagger(hyp)
# ref = pos_tagger(ref)

# for token in ref:
#     print(token.text, token.pos_, token.dep_)

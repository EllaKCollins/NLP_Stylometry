from datasets import load_dataset
import pandas as pd
import spacy
import re

#dataset = load_dataset("GroNLP/ik-nlp-22_pestyle", "full", data_dir="NLP_data")
#train_pd = dataset['train'].to_pandas()

## give the function the dataframe and it will return the substitutions, deletions and insertions to the dataframe
def retrieve_pos_from_dataframe(dataframe): 
    aligned_edits = dataframe["aligned_edit"]
    pos_tagger = spacy.load("it_core_news_sm")
    
    retrieved_subs = []
    retrieved_dels = []
    retrieved_ins = []

    for edit in aligned_edits: 
        if edit == "nan":
            retrieved_subs.append(edit)
            retrieved_dels.append(edit)
            retrieved_ins.append(edit)
        else:
            s_edit = edit.split("\\n")

            ref = s_edit[0].replace("REF:", "").replace("HYP:", "").replace("EVAL:", "")
            hyp = s_edit[1].replace("REF:", "").replace("HYP:", "").replace("EVAL:", "")

            ref = re.sub("\*+", "*", ref)
            hyp = re.sub("\*+", "*", hyp)

            ref = " ".join(re.split("\s+", re.sub("[^\w\d'*\s]+",'', ref), flags=re.UNICODE)).strip()
            hyp = " ".join(re.split("\s+", re.sub("[^\w\d'*\s]+",'', hyp), flags=re.UNICODE)).strip()

            ref_arr = ref.split()
            hyp_arr = hyp.split()
            hyp_evals = []
            for index, word in enumerate(ref_arr):
                if word == hyp_arr[index]:
                    hyp_evals.append("N")
                elif "*" in word: 
                    hyp_evals.append("I")
                elif "*" in hyp_arr[index]:
                    hyp_evals.append("D")
                else:
                    hyp_evals.append("S")

            ref_evals = hyp_evals.copy()
            for index, x in enumerate(ref_arr):
                if "'" in x: 
                    to_app = x.split("'")
                    ref_arr[index] = to_app[0]
                    ref_arr.insert(index+1, to_app[1])
                    ref_evals.insert(index+1, ref_evals[index])

            for index, x in enumerate(hyp_arr):
                if "'" in x: 
                    to_app = x.split("'")
                    hyp_arr[index] = to_app[0]
                    hyp_arr.insert(index+1, to_app[1])
                    hyp_evals.insert(index+1, hyp_evals[index])

            ref_pos = pos_tagger(ref)
            hyp_pos = pos_tagger(hyp)
            
            #print("ref: ", len(ref_evals), ", ", len(ref_pos))
            #print("hyp: ", len(hyp_evals), ", ", len(hyp_pos))
            if len(ref_evals) != len(ref_pos):
                print("ref")
                print(len(ref_evals), " ", len(ref_pos))
                print(ref)
                print(hyp)
                print(ref_evals)
                print([token for token in ref_pos])
                print([token.pos_ for token in ref_pos])

            if len(hyp_evals) != len(hyp_pos):
                print("hyp")
                print(len(hyp_evals), " ", len(hyp_pos))
                print(ref)
                print(hyp)
                print(hyp_evals)
                print([token for token in hyp_pos])
                print([token.pos_ for token in hyp_pos])

            
            for index, ev in enumerate(ref_evals):
                if ev == "S":
                    retrieved_subs.append(ref_pos[index].pos_)
                if ev == "D": 
                    retrieved_dels.append(ref_pos[index].pos_)

            for index, ev in enumerate(hyp_evals):
                if ev == "I": 
                    retrieved_ins.append(hyp_pos[index].pos_)
            if "I" not in hyp_evals:
                retrieved_ins.append("nan")
            if "S" not in ref_evals:
                retrieved_subs.append("nan")
            if "D" not in ref_evals:
                retrieved_dels.append("nan")

    dataframe["substitutions"] = retrieved_subs
    dataframe["deletions"] = retrieved_dels
    dataframe["insertions"] = retrieved_ins

    return dataframe


### NOT CORRECT YET
# the function will return the substitutions, deletions and insertions for a given edit as three objects.
def retrieve_edits_per_sentence(edit):
    if edit == "nan":
        return "nan", "nan", "nan"
    else:
        s_edit = edit.split("\\n")
        evals = s_edit[2]
        eval_S = [pos for pos, char in enumerate(evals) if char == "S"]
        eval_D = [pos for pos, char in enumerate(evals) if char == "D"]
        eval_I = [pos for pos, char in enumerate(evals) if char == "I"]
        if eval_S: # if substituted look at mt word -> REF
            temp = []
            for s in eval_S:
                temp.append(s_edit[0][s:].split()[0])
            retrieved_subs = temp
        else: 
            retrieved_subs = "nan"
        if eval_D: # if deleted find deleted word 
            temp = []
            for d in eval_D:
                temp.append(s_edit[0][d:].split()[0])
            retrieved_dels = temp
        else: 
            retrieved_dels = "nan"
        if eval_I: # if inserted look at inserted word 
            temp = []
            for i in eval_I:
                temp.append(s_edit[1][i:].split()[0])
            retrieved_ins = temp
        else: 
            retrieved_ins = "nan"
    return retrieved_subs, retrieved_dels, retrieved_ins

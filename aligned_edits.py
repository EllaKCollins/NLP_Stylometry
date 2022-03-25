from datasets import load_dataset
import pandas as pd

#dataset = load_dataset("GroNLP/ik-nlp-22_pestyle", "full", data_dir="NLP_data")
#train_pd = dataset['train'].to_pandas()

## give the function the dataframe and it will return the substitutions, deletions and insertions to the dataframe
def retrieve_edits_from_dataframe(dataframe): 
    aligned_edits = dataframe["aligned_edit"]

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
            evals = s_edit[2]
            eval_S = [pos for pos, char in enumerate(evals) if char == "S"]
            eval_D = [pos for pos, char in enumerate(evals) if char == "D"]
            eval_I = [pos for pos, char in enumerate(evals) if char == "I"]
            if eval_S: # if substituted look at mt word -> REF
                temp = []
                for s in eval_S:
                    temp.append(s_edit[0][s:].split()[0])
                retrieved_subs.append(temp)
            else: 
                retrieved_subs.append("nan")
            if eval_D: # if deleted find deleted word 
                temp = []
                for d in eval_D:
                    temp.append(s_edit[0][d:].split()[0])
                retrieved_dels.append(temp)
            else: 
                retrieved_dels.append("nan")
            if eval_I: # if inserted look at inserted word 
                temp = []
                for i in eval_I:
                    temp.append(s_edit[1][i:].split()[0])
                retrieved_ins.append(temp)
            else: 
                retrieved_ins.append("nan")

    dataframe["substitutions"] = retrieved_subs
    dataframe["deletions"] = retrieved_dels
    dataframe["insertions"] = retrieved_ins

    return dataframe

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

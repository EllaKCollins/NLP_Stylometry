from datasets import load_dataset
import pandas as pd
import spacy
from spacy.symbols import ORTH, LEMMA, POS
import re

#dataset = load_dataset("GroNLP/ik-nlp-22_pestyle", "full", data_dir="NLP_data")
#train_pd = dataset['train'].to_pandas()

def make_tagger():
    pos_tagger = spacy.load("it_core_news_sm")
    # PREPOSIZIONI ARTICOLATE
    # DI
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"della"}]], {LEMMA: u"della", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dello"}]], {LEMMA: u"dello", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"delle"}]], {LEMMA: u"delle", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"degli"}]], {LEMMA: u"degli", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dei"}]], {LEMMA: u"dei", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"del"}]], {LEMMA: u"del", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dell'"}]], {LEMMA: u"dell'", POS: u"ADP"})
    # A
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"alla"}]], {LEMMA: u"alla", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"allo"}]], {LEMMA: u"allo", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"alle"}]], {LEMMA: u"alle", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"agli"}]], {LEMMA: u"agli", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"ai"}]], {LEMMA: u"ai", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"al"}]], {LEMMA: u"al", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"all'"}]], {LEMMA: u"all'", POS: u"ADP"})
    # DA
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dalla"}]], {LEMMA: u"dalla", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dallo"}]], {LEMMA: u"dallo", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dalle"}]], {LEMMA: u"dalle", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dagli"}]], {LEMMA: u"dagli", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dai"}]], {LEMMA: u"dai", POS: u"ADP"}) # attention, ambiguité !
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dal"}]], {LEMMA: u"dal", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"dall'"}]], {LEMMA: u"dall'", POS: u"ADP"})
    # IN
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"nella"}]], {LEMMA: u"nella", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"nello"}]], {LEMMA: u"nello", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"nelle"}]], {LEMMA: u"nelle", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"negli"}]], {LEMMA: u"negli", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"nei"}]], {LEMMA: u"nei", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"nel"}]], {LEMMA: u"nel", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"nell'"}]], {LEMMA: u"nell'", POS: u"ADP"})
    # CON
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"coi"}]], {LEMMA: u"coi", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"col"}]], {LEMMA: u"col", POS: u"ADP"})
    # SU
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sulla"}]], {LEMMA: u"sulla", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sullo"}]], {LEMMA: u"sullo", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sulle"}]], {LEMMA: u"sulle", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sugli"}]], {LEMMA: u"sugli", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sui"}]], {LEMMA: u"sui", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sul"}]], {LEMMA: u"sul", POS: u"ADP"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"sull'"}]], {LEMMA: u"sull'", POS: u"ADP"})
    #
    # ARTCOLO DETERMINATIVO CON APOSTROFO
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"l'"}]], {LEMMA: u"l'", POS: u"DET"})
    pos_tagger.get_pipe("attribute_ruler").add([[{ORTH: u"l'"}]], {LEMMA: u"L'", POS: u"DET"})

    return pos_tagger


## give the function the dataframe and it will return the substitutions, deletions and insertions to the dataframe
def retrieve_pos_from_dataframe(dataframe): 
    pos_tagger = make_tagger()

    cols = list(dataframe.columns) + ["sub_POS", "del_POS", "in_POS"]
    dataframe_with_pos = pd.DataFrame(columns = cols)
    data_excluded = pd.DataFrame(columns = cols)

    for i, datapoint in dataframe.iterrows(): 
        edit = datapoint["aligned_edit"]
        if edit == "nan":
            dataframe_with_pos.loc[len(dataframe_with_pos)] = list(datapoint) + ["nan", "nan", "nan"]
        else:
            s_edit = edit.split("\\n") # split the ref, hyp and eval
            # remove beginnings
            ref = s_edit[0].replace("REF:", "").replace("HYP:", "").replace("EVAL:", "")
            hyp = s_edit[1].replace("REF:", "").replace("HYP:", "").replace("EVAL:", "")
            # shorten *** blocks
            ref = re.sub("\*+", "*", ref)
            hyp = re.sub("\*+", "*", hyp)
            # remove punctuation except * and ' and strip sentence of extra whitespace
            ref = " ".join(re.split("\s+", re.sub("[^\w\d'*\s]+",'', ref), flags=re.UNICODE)).strip()
            hyp = " ".join(re.split("\s+", re.sub("[^\w\d'*\s]+",'', hyp), flags=re.UNICODE)).strip()
            # create arrays of the sentences
            ref_arr = ref.split()
            hyp_arr = hyp.split()
            # determine the edits that were changed 
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
                    to_app = re.split("'", x)
                    ref_arr[index] = to_app[0]
                    for idx, item in enumerate(to_app[1:]):
                        if item != "":
                            ref_arr.insert(index+idx+1, item)
                            ref_evals.insert(index+idx+1, ref_evals[index])

            for index, x in enumerate(hyp_arr):
                if "'" in x: 
                    to_app = re.split("'", x)
                    hyp_arr[index] = to_app[0] # change item at position
                    for idx, item in enumerate(to_app[1:]):
                        if item != "":
                            hyp_arr.insert(index+idx+1, item)
                            hyp_evals.insert(index+idx+1, hyp_evals[index])

            # pos tagging the sentences
            ref_pos = pos_tagger(ref)
            hyp_pos = pos_tagger(hyp)

            # substitutions and deletions are checked in the ref sentence
            substitutions = []
            deletions = []
            if ("S" in ref_evals or "D" in ref_evals) and len(ref_evals) != len(ref_pos):
                data_excluded.loc[len(data_excluded)] = list(datapoint) + ["nan", "nan", "nan"]
                continue
            else:
                for index, ev in enumerate(ref_evals):
                    if ev == "S":
                        substitutions.append(ref_pos[index].pos_)
                    if ev == "D": 
                        deletions.append(ref_pos[index].pos_)
            

            # insertions are checked in the hyp sentence
            insertions = []
            if "I" in hyp_evals and len(hyp_evals) != len(hyp_pos):
                data_excluded.loc[len(data_excluded)] = list(datapoint) + ["nan", "nan", "nan"]
                continue
            else:
                for index, ev in enumerate(hyp_evals):
                    if ev == "I":
                        insertions.append(hyp_pos[index].pos_)

            dataframe_with_pos.loc[len(dataframe_with_pos)] = list(datapoint) + [substitutions, deletions, insertions]

    return dataframe_with_pos, data_excluded
import pandas as pd
from collections import Counter

def convert_pos_to_counts(dataframe):

    columns = ['item_id', 'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'EOL', 'SPACE']
    pos_dataframe = pd.DataFrame(columns = columns)

    for i, datapoint in dataframe.iterrows():
        if datapoint["aligned_edit"] != "nan":
            pos = datapoint["sub_POS"] + datapoint["del_POS"] + datapoint["in_POS"]
            pos_dict = Counter(pos)
            array = [datapoint["item_id"]] + [0] * 20
            for key in pos_dict:
                array[columns.index(key)] = pos_dict[key]
            pos_dataframe.loc[len(pos_dataframe)] = array

    return pos_dataframe

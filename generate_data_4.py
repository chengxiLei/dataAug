from datasets import load_dataset, load_metric
from datasets import Audio
from datasets import Features
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
import wave

import re

#tokenizer
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

timit = load_dataset("timit_asr", data_dir='./');
#delete irrelavent info
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
timit = timit.map(remove_special_characters)
print(timit["train"].features)

import torch
import numpy as np
nfft = 1024
# change the dataset
from phaseAug import harmonics
from phaseAug import phase_function


def mapFunc(e,id):
    # e["text"] = "GOGOGOGOGO"
    print(id,"is finished")
    X = torch.stft(
                    torch.as_tensor(e["audio"]["array"]),
                    nfft,
                    256,
                    window= torch.hann_window(nfft),
                    return_complex=True
                )
    #complex aug
    for tBinN in range(0,X.shape[1]-1):
        if tBinN%1 == 0 :
            X = harmonics(tBinN,X,nfft)
        for wBinN in range(0,X.shape[0]-1):
            randNum = np.random.randn()
            X[wBinN][tBinN] = X[wBinN][tBinN] * phase_function(tBinN,X.shape[1]-1,wBinN,X) 
            # X[wBinN][tBinN] = X[wBinN][tBinN] * amp_function(tBinN,X.shape[1]-1) 
            X[wBinN][tBinN] = X[wBinN][tBinN] * randNum * randNum 

    x_aug = torch.istft(
                    X,
                    nfft,
                    256,
                    window=torch.hann_window(nfft),
                    return_complex=False
                )
    e["audio"]["array"] = np.asarray(x_aug)

    
    return e

from datasets import concatenate_datasets
arr = list(range(3000,4000))
generated_data = timit["train"].select(arr).map(mapFunc,with_indices=True)
# new_set = concatenate_datasets([generated_data,timit["train"]])
import datetime 
dt = datetime.datetime.now() # use now() method in datetime
strName = "./new_data_" + str(dt.month) + "_" + str(dt.day) + "/4"
generated_data.save_to_disk(strName)

print("done")






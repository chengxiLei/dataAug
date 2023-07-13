import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10,10,200)
y = np.sin(x)
plt.figure(figsize=(2,1))
plt.subplot()
plt.plot(x,y)

import datetime 
dt = datetime.datetime.now() # use now() method in datetime

strName = "./new_data_" + str(dt.month) + "_" + str(dt.day) + "/1"
print( "Display the current date of the system: ") # current date 
print (strName) # call the dt variable to print the system date.

from datasets import load_dataset, load_metric

from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

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

timit = load_dataset("Siyong/speech_timit", data_dir='./');
#delete irrelavent info
# timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
# timit = timit.map(remove_special_characters)

import IPython.display as ipd
import numpy as np
import random


# from matplotlib import pyplot as plt
rand_int = random.randint(0, len(timit["train"]))
rand_int = 20


arr = list(range(4000,4620))
#how to access the data
print(arr)
# print("Input array shape:", np.asarray(timit["train"][rand_int]["audio"]["array"]).shape)
# print("Sampling rate:", timit["train"][rand_int]["audio"]["sampling_rate"])
ipd.Audio(data=np.asarray(timit["train"][rand_int]["audio"]["array"]), autoplay=True, rate=16000)
# print(timit["train"].features)

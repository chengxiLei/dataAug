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

timit = load_dataset("timit_asr", data_dir='./');
#delete irrelavent info
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
timit = timit.map(remove_special_characters)
show_random_elements(timit["train"].remove_columns(["file", "audio"]))
#statistic of alphabet
vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# ##feature extractor

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

import IPython.display as ipd
import numpy as np
import random


# from matplotlib import pyplot as plt
rand_int = random.randint(0, len(timit["train"]))
rand_int = 20



#how to access the data
print("Target text:", timit["train"][rand_int]["text"])
# print("Input array shape:", np.asarray(timit["train"][rand_int]["audio"]["array"]).shape)
# print("Sampling rate:", timit["train"][rand_int]["audio"]["sampling_rate"])
# ipd.Audio(data=np.asarray(timit["train"][rand_int]["audio"]["array"]), autoplay=True, rate=16000)
print(timit)


import torch
nfft = 1024
X = torch.stft(
                    torch.as_tensor(timit["train"][rand_int]["audio"]["array"]),
                    nfft,
                    256,
                    window= torch.hann_window(nfft),
                    return_complex=True
                )

# print(X.shape[0])# frequency
# print(X.shape[1])# time
import numpy as np
def amp_function (tBinN, length):
    # return 3*np.sin(((np.pi/2)/length)*tBinN)
    k = 0.5/500
    return 0

def phase_function (tBinN, length, wBinN):
    randNum = np.random.randn()
    return torch.exp(torch.tensor([(0.+1.j)], dtype = torch.complex64, ) * (6.28*1/(X.shape[1] + 1))*1*(0.3*tBinN*wBinN*randNum*randNum+1))
# methods 
# def phase_function (tBinN, length,wBinN):
#     return torch.exp(torch.tensor([(0.+1.j)], dtype = torch.complex64, ) * (6.28/(X.shape[1] + 1))*1*(wBinN*tBinN+1)) 
# def phase_function (tBinN, length,wBinN):
#     # return torch.exp(torch.tensor([(0.+1.j)], dtype = torch.complex64, ) * (6.28/(X.shape[1] + 1))*1*(tBinN+1)) 
#     return torch.exp(torch.tensor([(0.+1.j)], dtype = torch.complex64, ) * (6.28/(X.shape[1] + 1))*1*(0.25*tBinN*tBinN+wBinN*tBinN*0.1+1)) 

def harmonics (tBinN):
    randNum = np.random.randn()
    # randNum = 1
    scaler = 1.4
    width = 10
    mainFreq = np.argmax(X[:][tBinN])
    if mainFreq*2 <= (nfft/2) :
        for i in range(0,width):
            if (mainFreq*2+i) <= (nfft/2 - 5) :
                X[mainFreq*2+i][tBinN] = X[mainFreq*2+i][tBinN]*scaler*randNum
                X[mainFreq*2-i][tBinN] = X[mainFreq*2-i][tBinN]*scaler*randNum
    if mainFreq*3 <= (nfft/2) :
        for i in range(0,width):
            if (mainFreq*3+i) <= (nfft/2 - 5) :
                X[mainFreq*3+i][tBinN] = X[mainFreq*3+i][tBinN]*(scaler/2)*randNum
                X[mainFreq*3-i][tBinN] = X[mainFreq*3-i][tBinN]*(scaler/2)*randNum
    

    # if mainFreq*3 <= (nfft/2) :
    #     X[mainFreq*3][tBinN] = X[mainFreq][tBinN]*1.2
    # if mainFreq*4 <= (nfft/2) :
    #     X[mainFreq*4][tBinN] = X[mainFreq][tBinN]*1.2
    # if mainFreq*5 <= (nfft/2) :
    #     X[mainFreq*5][tBinN] = X[mainFreq][tBinN]*1.2

#original spect
# spectrArrOri = [] 
# for tBinN in range(0,X.shape[1]-1):
#     spectrArrOri.append(np.argmax(X[:][tBinN]))

#complex aug
for tBinN in range(0,X.shape[1]-1):
    if tBinN%1 == 0 :
        harmonics(tBinN)
    for wBinN in range(0,X.shape[0]-1):
        if tBinN%1 == 0 :
            randNum = np.random.randn()
            X[wBinN][tBinN] = X[wBinN][tBinN] * phase_function(tBinN,X.shape[1]-1,wBinN) 
            # X[wBinN][tBinN] = X[wBinN][tBinN] * amp_function(tBinN,X.shape[1]-1) 
            X[wBinN][tBinN] = X[wBinN][tBinN] * randNum * randNum  

#auged spect
# spectrArrNew = [] 
# for tBinN in range(0,X.shape[1]-1):
#     spectrArrNew.append(np.argmax(X[:][tBinN]))


x_aug = torch.istft(
                    X,
                    nfft,
                    256,
                    window=torch.hann_window(nfft),
                    return_complex=False
                )
# print(timit["train"][rand_int]["audio"]["array"])
# print(len(timit["train"][rand_int]["audio"]["array"]))
# print(len(np.asarray(x_aug)))

# print(type(timit))
# print(type(timit["train"][rand_int]["audio"]["array"][2]))

# print(type(np.asarray(x_aug)[2]))
# timit["train"][rand_int]["audio"]["array"][2] = np.asarray(x_aug)[2]

# for n1 in range(0,len(np.asarray(x_aug))):
#     timit["train"][rand_int]["audio"]["array"][n1] = np.asarray(x_aug)[n1]

# print(timit["train"][rand_int]["audio"]["array"][2])
# print(np.asarray(x_aug).shape)
# print(len(np.asarray(x_aug)))

#reverse
# for i in range(0,len(np.asarray(x_aug))):
#     np.asarray(x_aug)[i] = -np.asarray(x_aug)[i]

import matplotlib.pyplot as plt
import numpy as np
startNum = 37400
cutNum = 37800
xAix1 = np.linspace(1,len(np.asarray(x_aug)[startNum:cutNum]),len(np.asarray(x_aug)[startNum:cutNum]))
yAix1 = np.asarray(x_aug)[startNum:cutNum]
xAix2 = np.linspace(1,len(np.asarray(timit["train"][rand_int]["audio"]["array"])[startNum:cutNum]),len(np.asarray(timit["train"][rand_int]["audio"]["array"])[startNum:cutNum]))
yAix2 = np.asarray(timit["train"][rand_int]["audio"]["array"])[startNum:cutNum]
plt.figure()
plt.subplot(311)
plt.plot(xAix1,yAix1)
plt.plot(xAix2,yAix2)
plt.subplot(312)
plt.plot(xAix1,yAix1)
plt.subplot(313)
plt.plot(xAix2,yAix2)

# # xAix3 = np.linspace(1,X.shape[1]-1,X.shape[1]-1)
# # plt.subplot(413)
# # plt.plot(xAix3,spectrArrOri)
# # plt.subplot(414)
# # plt.plot(xAix3,spectrArrNew)
# ipd.Audio(data=np.asarray(timit["train"][rand_int]["audio"]["array"]), autoplay=True, rate=16000)
ipd.Audio(data=np.asarray(x_aug), autoplay=True, rate=16000)

# import librosa
# # mel_spect = librosa.feature.melspectrogram(y=np.asarray(x_aug), sr=16000, n_fft=1024, hop_length=512)
# # mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
# # librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
# # plt.title('Mel Spectrogram');
# # plt.colorbar(format='%+2.0f dB');

# mel_spect2 = librosa.feature.melspectrogram(y=np.asarray(timit["train"][rand_int]["audio"]["array"]), sr=16000, n_fft=1024, hop_length=512)
# mel_spect2 = librosa.power_to_db(mel_spect2, ref=np.max)
# librosa.display.specshow(mel_spect2, y_axis='mel', fmax=8000, x_axis='time');
# plt.title('Mel Spectrogram2');
# plt.colorbar(format='%+2.0f dB');






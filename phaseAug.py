import torch
import numpy as np
# change the dataset
def phase_function (tBinN, length,wBinN,X):
    randNum = np.random.randn()
    return torch.exp(torch.tensor([(0.+1.j)], dtype = torch.complex64, ) * (6.28/(X.shape[1] + 1))*1*(1*wBinN*randNum*randNum+1))

def harmonics (tBinN,X,nfft):
    randNum = np.random.randn()
    scaler = 1.2
    width = 10
    mainFreq = np.argmax(X[:][tBinN])
    if mainFreq*2 <= (nfft/2 - 5) :
        for i in range(0,width):
            if (mainFreq*2+i) <= (nfft/2 - 5) :
                X[mainFreq*2+i][tBinN] = X[mainFreq*2+i][tBinN]*scaler*randNum
                X[mainFreq*2-i][tBinN] = X[mainFreq*2-i][tBinN]*scaler*randNum
    if mainFreq*3 <= (nfft/2) :
        for i in range(0,width):
            if (mainFreq*3+i) <= (nfft/2 - 5) :
                X[mainFreq*3+i][tBinN] = X[mainFreq*3+i][tBinN]*(scaler/2)*randNum
                X[mainFreq*3-i][tBinN] = X[mainFreq*3-i][tBinN]*(scaler/2)*randNum
    return X


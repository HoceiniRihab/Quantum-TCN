import numpy as np
import torch
from scipy.io import loadmat


def data_generator(dataset):
    
    if dataset == "JSB":
        print('Loading JSB Chorales data...')
        data = loadmat('./mdata/JSB_Chorales.mat')
    elif dataset == "Muse":
        print('Loading MuseData...')
        data = loadmat('./mdata/MuseData.mat')
    elif dataset == "Nott":
        print('Loading Nottingham data...')
        data = loadmat('./mdata/Nottingham.mat')
    elif dataset == "Piano":
        print('Loading Piano-midi data...')
        data = loadmat('./mdata/Piano_midi.mat')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]
    for data_split in [X_train, X_valid, X_test]:
        for i in range(len(data_split)):
            data_split[i] = torch.Tensor(data_split[i].astype(np.float64))
    
    return X_train, X_valid, X_test


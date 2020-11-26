import os
from os import path
import numpy as np
import pickle
import itertools

from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew

local_directory = "E:\\bjark\\Desktop\\dataanalyse\\Data\\genres_original\\"
fileName = "features.dat"

def get_statistics(desc):
    result = {}
    
    for k, values in desc.items():
        if k != 'fileName':
            # Ugly hack to create the dictionary
            result['{}_max'.format(k)] = []
            result['{}_min'.format(k)] = []
            result['{}_mean'.format(k)] = []
            result['{}_std'.format(k)] = []
            result['{}_kurtosis'.format(k)] = []
            result['{}_skew'.format(k)] = []
            for v in values:
                result['{}_max'.format(k)].append(np.max(v))
                result['{}_min'.format(k)].append(np.min(v))
                result['{}_mean'.format(k)].append(np.mean(v))
                result['{}_std'.format(k)].append(np.std(v))
                result['{}_kurtosis'.format(k)].append(kurtosis(v))
                result['{}_skew'.format(k)].append(skew(v))
    return result

# For feature extraction we use Mel Frequency Cepstral Coefficients (MFCC)
# This makes the data more concise, and reduces features per frame.
def build_gtzan(path, f):

    # Features to build in the final dictionary
    features = {'fileName': [], 'mfcc_mean_matrix': [], 'variance': []}

    folderIndex = 0    
    """Load GTZAN data from `path`"""
    for folder in os.listdir(path):

        # We have 10 different genres
        folderIndex += 1
        if folderIndex == 11:
            break

        print("[",folderIndex,"/",10,"] Folders processed...")
        
        for file in os.listdir(path+folder):
            y, sr = librosa.load(path+folder+"/"+file)
            # MFCC treatment
            mfcc  = librosa.feature.mfcc(y, sr, n_mfcc=13)
            variance = mfcc.var(0)
            mean_matrix = mfcc.mean(0)
            features['fileName'].append(file)
            features['mfcc_mean_matrix'].append(mean_matrix)
            features['variance'].append(variance)

    dict_features = get_statistics(features)
    dict_features['fileName'] = features['fileName']
    df = pd.DataFrame(dict_features)
    pickle.dump(df, f)

# Used for compression features to a binary file.            
def build_binaryfile():
    if(path.exists(fileName) == False):
        f = open(fileName, 'wb')
        build_gtzan(local_directory, f)
        f.close()

def load_binaryfile(file):

    data = None
    
    with open(fileName, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                f.close()
                break

    df = pd.DataFrame(data)
    print(df)

def main():
    """Try building the binary file of features. If it exists don't do anything"""
    build_binaryfile()
    load_binaryfile(fileName)

if __name__== "__main__":
    main()
    

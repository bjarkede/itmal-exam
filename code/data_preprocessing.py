import os
from os import path
import numpy as np
import pickle
import itertools
from time import time

import librosa
import librosa.display
import pandas as pd

from scipy.stats import kurtosis
from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

local_directory = "C:\\Users\\jonas\\Desktop\\itmal-exame\\itmal-exam\\code\\"
fileName = "features.dat"

# Gets various statistics for each feature in the desc
def get_statistics(desc):
    result = {}
    
    for k, values in desc.items():
        if k != 'classlabel':
            # Ugly hack to create the dictionary
            result['{}_max'.format(k)] = []
            result['{}_min'.format(k)] = []
            result['{}_mean'.format(k)] = []
            #result['{}_std'.format(k)] = []
            #result['{}_kurtosis'.format(k)] = []
            #result['{}_skew'.format(k)] = []
            for v in values:
                result['{}_max'.format(k)].append(np.max(v))
                result['{}_min'.format(k)].append(np.min(v))
                result['{}_mean'.format(k)].append(np.mean(v))
                #result['{}_std'.format(k)].append(np.std(v))
                #result['{}_kurtosis'.format(k)].append(kurtosis(v))
                #result['{}_skew'.format(k)].append(skew(v))
    return result

# For feature extraction we use Mel Frequency Cepstral Coefficients (MFCC)
# This makes the data more concise, and reduces features per frame.
def build_gtzan(path, f):

    # Features to build in the final dictionary
    features = {'classlabel': [], 'chroma_stft_mean':[],'chroma_stft_var':[],'rms_mean':[],
           'rms_var':[],'bandwidth_mean':[],'bandwidth_var':[],'centroid_mean':[],
           'centroid_var':[],'zero_crossing_mean':[],'zero_crossing_var':[],
           'harmony_mean':[],'harmony_var':[],'rolloff_mean':[],'rolloff_var':[],
           'perceptrual_mean':[],'perceptrual_var':[],'tempo':[]}

    for i in range(20):
        feat1 = f"mfcc{i+1}_mean"
        feat2 = f"mfcc{i+1}_var"
        features[feat1] = []
        features[feat2] = []
        
    start = time()
    print("\n")
    folderIndex = 0    
    """Load GTZAN data from `path`"""
    for folder in os.listdir(path):

        # We have 10 different genres
        folderIndex += 1
        if folderIndex == 11:
            break

        print("[",folderIndex,"/",10,"] Folders processed...")
        
        for file in os.listdir(path+folder):
            fileindex = 0
            for i in range(10):
                dur = 3.0 * fileindex
                y, sr = librosa.load(path+folder+"/"+file,offset=dur,duration=3.0)
                fileindex += 1
                #Removing silence
                y_trimmed,_ = librosa.effects.trim(y)
                
                #zero crossing
                zeroX = librosa.feature.zero_crossing_rate(y_trimmed,pad=False)
                zeroX_mean = zeroX.mean()
                zeroX_var = zeroX.var()
                
                
                #harmony & perceptrual
                harm, perc = librosa.effects.hpss(y_trimmed)
                harm_mean = harm.mean()
                harm_var = harm.var()
                perc_mean = perc.mean()
                perc_var = perc.var()
                
                #Tempo
                tempo,_ = librosa.beat.beat_track(y=y,sr=sr)
                
                #Centroid
                centroid = librosa.feature.spectral_centroid(y=y_trimmed,sr=sr)[0]
                centroid_mean = centroid.mean()
                centroid_var = centroid.var()
                
                #Rolloff
                rolloff = librosa.feature.spectral_rolloff(y=y_trimmed,sr=sr)[0]
                rolloff_mean = rolloff.mean()
                rolloff_var = rolloff.var()
                
                #Chroma frequencies
                chroma = librosa.feature.chroma_stft(y=y_trimmed,sr=sr)
                chroma_mean = chroma.mean()
                chroma_var = chroma.var()
            
                #RMS
                rms = librosa.feature.rms(y)
                rms_mean = rms.mean()
                rms_var = rms.var()
            
                #spectral bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(y,sr)
                bandwidth_mean = bandwidth.mean()
                bandwidth_var = bandwidth.var()
            

            
            # MFCC treatment
                for i in range(20):
                    mfcc  = librosa.feature.mfcc(y, sr, n_mfcc=i+1)
                    mfcc_mean = mfcc.mean()
                    mfcc_var = mfcc.var()
                    feat1 = f"mfcc{i+1}_mean"
                    feat2 = f"mfcc{i+1}_var"
                    features[feat1].append(np.mean(mfcc_mean))
                    features[feat2].append(np.mean(mfcc_var))
            

                
                
                #adding features
                features['classlabel'].append(str(file).split('.', 1)[0])
                features['zero_crossing_mean'].append(zeroX_mean)
                features['zero_crossing_var'].append(zeroX_var)
                features['harmony_mean'].append(harm_mean)
                features['harmony_var'].append(harm_var)
                features['perceptrual_mean'].append(perc_mean)
                features['perceptrual_var'].append(perc_var)
                features['tempo'].append(tempo)
                features['centroid_mean'].append(centroid_mean)
                features['centroid_var'].append(centroid_var)
                features['rolloff_mean'].append(rolloff_mean)
                features['rolloff_var'].append(rolloff_mean)
                features['chroma_stft_mean'].append(chroma_mean)
                features['chroma_stft_var'].append(chroma_var)
                features['rms_mean'].append(rms_mean)
                features['rms_var'].append(rms_var)
                features['bandwidth_mean'].append(bandwidth_mean)
                features['bandwidth_var'].append(bandwidth_var)



    #dict_features = get_statistics(features)
    #dict_features['classlabel'] = features['classlabel']
    #df = pd.DataFrame(dict_features)
    df = pd.DataFrame(features)
    pickle.dump(df, f)
    t = (time() - start) / 60
    print((t))
        
# Used for compression features to a binary file.            
def build_binaryfile(pathName):
    f = open(fileName, 'wb')
    build_gtzan(pathName, f)
    f.close()

# Used for loading a .dat binary file and loading it into a pandas DataFrame.
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
    return df

# Used for encoding genre labels as integers
def encode_class_labels(df):
    class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
    df['classlabel'] = df['classlabel'].map(class_mapping)

# Partion the dataset into seperate train/test sets.
# Furthermore we use standardization for optimization for algorithms such as gradiant descent.
def partion_dataset(df):
    stdsc = StandardScaler()
    
    X, y = df.iloc[:, 0:-1].values, df.iloc[:,-1:].values
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return (stdsc.fit_transform(X_train), stdsc.transform(X_test), y_train, y_test)

def get_features(pathtofile):
    y, sr = librosa.load(pathtofile,duration=30)

    #Removing silence
    y_trimmed,_ = librosa.effects.trim(y)
                
    #zero crossing
    zeroX = librosa.feature.zero_crossing_rate(y_trimmed,pad=False)
    zeroX_mean = zeroX.mean()
    zeroX_var = zeroX.var()
                
                
    #harmony & perceptrual
    harm, perc = librosa.effects.hpss(y_trimmed)
    harm_mean = harm.mean()
    harm_var = harm.var()
    perc_mean = perc.mean()
    perc_var = perc.var()
                
    #Tempo
    tempo,_ = librosa.beat.beat_track(y=y,sr=sr)
               
    #Centroid
    centroid = librosa.feature.spectral_centroid(y=y_trimmed,sr=sr)[0]
    centroid_mean = centroid.mean()
    centroid_var = centroid.var()
                
    #Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y_trimmed,sr=sr)[0]
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()
                
    #Chroma frequencies
    chroma = librosa.feature.chroma_stft(y=y_trimmed,sr=sr)
    chroma_mean = chroma.mean()
    chroma_var = chroma.var()
            
    #RMS
    rms = librosa.feature.rms(y)
    rms_mean = rms.mean()
    rms_var = rms.var()
            
    #spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y,sr)
    bandwidth_mean = bandwidth.mean()
    bandwidth_var = bandwidth.var()
            

            
    # MFCC treatment
    for i in range(20):
        mfcc  = librosa.feature.mfcc(y, sr, n_mfcc=i+1)
        mfcc_mean = mfcc.mean()
        mfcc_var = mfcc.var()
        feat1 = f"mfcc{i+1}_mean"
        feat2 = f"mfcc{i+1}_var"
        features[feat1].append(np.mean(mfcc_mean))
        features[feat2].append(np.mean(mfcc_var))
            

                
                
        #adding features
    features['classlabel'].append("")
    features['zero_crossing_mean'].append(zeroX_mean)
    features['zero_crossing_var'].append(zeroX_var)
    features['harmony_mean'].append(harm_mean)
    features['harmony_var'].append(harm_var)
    features['perceptrual_mean'].append(perc_mean)
    features['perceptrual_var'].append(perc_var)
    features['tempo'].append(tempo)
    features['centroid_mean'].append(centroid_mean)
    features['centroid_var'].append(centroid_var)
    features['rolloff_mean'].append(rolloff_mean)
    features['rolloff_var'].append(rolloff_mean)
    features['chroma_stft_mean'].append(chroma_mean)
    features['chroma_stft_var'].append(chroma_var)
    features['rms_mean'].append(rms_mean)
    features['rms_var'].append(rms_var)
    features['bandwidth_mean'].append(bandwidth_mean)
    features['bandwidth_var'].append(bandwidth_var)
        
        
    df = pd.DataFrame(features)
    return df

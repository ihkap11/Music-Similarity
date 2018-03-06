
# coding: utf-8

# In[4]:


import os
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    return columns.sort_values()



def compute_features(base_path):
      
    final = pd.DataFrame(index=[1],columns=columns(), dtype=np.float32)
    
    def sub_features(name, values):
            features[name, 'mean'] = np.mean(values, axis=1)
            features[name, 'std'] = np.std(values, axis=1)
            features[name, 'skew'] = stats.skew(values, axis=1)
            features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
            features[name, 'median'] = np.median(values, axis=1)
            features[name, 'min'] = np.min(values, axis=1)
            features[name, 'max'] = np.max(values, axis=1)
            
    warnings.filterwarnings('error', module='librosa')
            
    try:    
        for root, dirnames, filenames in os.walk(base_path):
                row = 1
                for filename in filenames: 
                    print(filename)
                    filepath=os.path.join(root, filename ).replace('\\', '/')

                    features = pd.Series(index=columns(), dtype=np.float32)


                    x, sr = librosa.load(filepath, sr=None, mono=True)

    #             """ZERO CROSSING RATE 
    #                 The rate at which the signal changes from positive to negative or back.
    #                 Key feature to classify percussive sounds.
    #                 Percussion sounds: resonant, hyper-resonant, stony dull or dull.
    #             """
                    zcr = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
                    sub_features("zcr", zcr)

                # 
                #    Losely relates to the twelve different pitch classes.

                #    CHROMA CQT - Constant Q
                #    It converts data series to frequency domain

                #    CHROMA CENS

                # """
                    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))
                    assert cqt.shape[0] == 7 * 12
                    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

                    chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
                    sub_features('chroma_cqt', chroma_cqt)

                    chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
                    sub_features('chroma_cens', chroma_cens)

                    del cqt

                # """TONNETZ

                # """
                    tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)
                    sub_features('tonnetz', tonnetz)

                # """ stft
                # """
                    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
                    assert stft.shape[0] == 1 + 2048 // 2
                    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
                    del x


                    chroma_stft = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
                    sub_features('chroma_stft', chroma_stft)

                    rmse = librosa.feature.rmse(S=stft)
                    sub_features('rmse', rmse)

                    f = librosa.feature.spectral_centroid(S=stft)
                    sub_features('spectral_centroid', f)
                    f = librosa.feature.spectral_bandwidth(S=stft)
                    sub_features('spectral_bandwidth', f)
                    f= librosa.feature.spectral_contrast(S=stft, n_bands=6)
                    sub_features('spectral_contrast', f)
                    f = librosa.feature.spectral_rolloff(S=stft)
                    sub_features('spectral_rolloff', f)

                    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
                    del stft
                    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
                    sub_features('mfcc', f)

                    final.loc[row] = features
    #                 print(final)
    #                 print("ROW:::",row)
                    row+=1
    except Exception as e:
        print('ERROR! Check path to the audio folder.')
    
    return final

        

def echonest_feature_maker(path,name):   
    df = compute_features(path)
    df.to_csv(name+'.csv')

    
    
    
"""echonest_feature_maker("C:/Users/spari/Projects/DM Pro/fma-master/raw","hi")"""





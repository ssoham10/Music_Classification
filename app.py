import streamlit as st
import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


pickle_in = open("knn.pkl", 'rb')
knn = pickle.load(pickle_in)

def genre_predict(df):
    prediction = knn.predict(df)
    print(prediction)
    return prediction

def transform_wav_to_csv(sound_saved):
    y, sr = librosa.load(sound_saved)
    # Get RMS value from each frame's magnitude value
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    #zcrs
    zcrs = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zcrs)
    zero_crossing_rate_var = np.var(zcrs)

    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chromagram)
    chroma_stft_var = np.var(chromagram)

    #tempo
    hop_length = 512
    # Compute local onset autocorrelation
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Estimate the global tempo for display purposes
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,hop_length=hop_length)[0]

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_var = np.var(spectral_rolloff)

    #spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    #harmony
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    #perceptr
    perceptr = librosa.effects.percussive(y)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)
    #mfcc
    mfcc = librosa.feature.mfcc(y)
    mfcc1 = mfcc[0]
    mfcc1_mean = np.mean(mfcc1)
    mfcc1_var = np.var(mfcc1)
    mfcc2 = mfcc[1]
    mfcc2_mean = np.mean(mfcc2)
    mfcc2_var = np.var(mfcc2)
    mfcc3 = mfcc[2]
    mfcc3_mean = np.mean(mfcc3)
    mfcc3_var = np.var(mfcc3)
    mfcc4 = mfcc[3]
    mfcc4_mean = np.mean(mfcc4)
    mfcc4_var = np.var(mfcc4)
    mfcc5 = mfcc[4]
    mfcc5_mean = np.mean(mfcc5)
    mfcc5_var = np.var(mfcc5)
    mfcc6 = mfcc[5]
    mfcc6_mean = np.mean(mfcc6)
    mfcc6_var = np.var(mfcc6)
    mfcc7 = mfcc[6]
    mfcc7_mean = np.mean(mfcc7)
    mfcc7_var = np.var(mfcc7)
    mfcc8 = mfcc[7]
    mfcc8_mean = np.mean(mfcc8)
    mfcc8_var = np.var(mfcc8)
    mfcc9 = mfcc[8]
    mfcc9_mean = np.mean(mfcc9)
    mfcc9_var = np.var(mfcc9)
    mfcc10 = mfcc[9]
    mfcc10_mean = np.mean(mfcc10)
    mfcc10_var = np.var(mfcc10)
    mfcc11 = mfcc[10]
    mfcc11_mean = np.mean(mfcc11)
    mfcc11_var = np.var(mfcc11)
    mfcc12 = mfcc[11]
    mfcc12_mean = np.mean(mfcc12)
    mfcc12_var = np.var(mfcc12)
    mfcc13 = mfcc[12]
    mfcc13_mean = np.mean(mfcc13)
    mfcc13_var = np.var(mfcc13)
    mfcc14 = mfcc[13]
    mfcc14_mean = np.mean(mfcc14)
    mfcc14_var = np.var(mfcc14)
    mfcc15 = mfcc[14]
    mfcc15_mean = np.mean(mfcc15)
    mfcc15_var = np.var(mfcc15)
    mfcc16 = mfcc[15]
    mfcc16_mean = np.mean(mfcc16)
    mfcc16_var = np.var(mfcc16)
    mfcc17 = mfcc[16]
    mfcc17_mean = np.mean(mfcc17)
    mfcc17_var = np.var(mfcc17)
    mfcc18 = mfcc[17]
    mfcc18_mean = np.mean(mfcc18)
    mfcc18_var = np.var(mfcc18)
    mfcc19 = mfcc[18]
    mfcc19_mean = np.mean(mfcc19)
    mfcc19_var = np.var(mfcc19)
    mfcc20 = mfcc[19]
    mfcc20_mean = np.mean(mfcc20)
    mfcc20_var = np.var(mfcc20)

    from sorcery import dict_of
    d = dict_of( chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean,
                 spectral_bandwidth_var, rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var,
                 perceptr_mean, perceptr_var, tempo, mfcc1_mean, mfcc1_var, mfcc2_mean, mfcc2_var, mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var,
                 mfcc5_mean, mfcc5_var, mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var, mfcc9_mean, mfcc9_var, mfcc10_mean,
                 mfcc10_var, mfcc11_mean, mfcc11_var, mfcc12_mean, mfcc12_var, mfcc13_mean, mfcc13_var, mfcc14_mean, mfcc14_var, mfcc15_mean,
                 mfcc15_var, mfcc16_mean, mfcc16_var, mfcc17_mean, mfcc17_var, mfcc18_mean, mfcc18_var, mfcc19_mean, mfcc19_var, mfcc20_mean, mfcc20_var)

    input_df = pd.DataFrame([d])

    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_df)
    return scaled_input

def main():
    st.title('Music Genre Classification')

    input_audio = st.file_uploader(label='Upload a .wav file', type=['wav'])
    if st.button(label='Check the Genre'):
        input_df = transform_wav_to_csv(input_audio)
        result = genre_predict(input_df)
        st.success("The Genre is {}.".format(result[0]))

if __name__ =="__main__":
    main()



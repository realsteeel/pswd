import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft
import streamlit as st

def process_ecg():
    data = loadmat('file/ecg.mat')  # Load ECG data from a .mat file
    ecg = data['ecg'].flatten()  # Assuming the variable name in .mat is 'ecg'
    
    Fs = st.number_input("Frekuensi Sampling (Hz):", min_value=1, value=500)
    G = st.number_input("Skala Amplitudo:", value=1.0)

    if st.button("Proses Sinyal ECG"):
        ecg = ecg / G
        ecg = (ecg - np.mean(ecg)) / np.std(ecg)
        t = np.arange(0, len(ecg)) * (1 / Fs)

        # Time-domain plot
        plt.figure(1)
        plt.plot(t, ecg)
        plt.xlim([0, 4])
        plt.xlabel('Waktu (s)')
        plt.ylabel('Amplitudo (mV)')
        plt.title('ECG dalam Domain Waktu')
        plt.grid(True)
        st.pyplot(plt)

        # Frequency-domain plot
        F = np.abs(fft(ecg))[:len(ecg)//2]
        F /= max(F)
        L = len(F)
        f = np.linspace(0, Fs/2, L)

        plt.figure(2)
        plt.plot(f, F)
        plt.xlabel('Frekuensi (Hz)')
        plt.ylabel('Magnitude Normalisasi')
        plt.title('ECG dalam Frekuensi')
        plt.grid(True)
        st.pyplot(plt)

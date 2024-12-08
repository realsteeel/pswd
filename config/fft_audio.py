import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import streamlit as st

def plot_fft():
    fs, bintangkecil = wavfile.read('file\Bintang Kecil.wav')

    F = np.fft.fft(bintangkecil)
    F = np.abs(F[:len(F)//2])
    F /= max(F)

    L = len(F)
    f = np.linspace(0, fs/2, L)

    plt.plot(f, F)
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Magnitude Normalisasi')
    plt.title('Spektrum Frekuensi dari audio')
    plt.grid(True)
    st.pyplot(plt)

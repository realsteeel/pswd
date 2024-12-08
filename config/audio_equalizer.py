import numpy as np
import streamlit as st
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def equalizer(audio, fs, g1, g2, g3, volume):
    N = 15  # Use an odd number of coefficients
    fc1 = 1500  # Low cutoff frequency
    fc2 = 5000  # High cutoff frequency

    # Normalized cutoff frequencies
    wc1 = fc1 / (fs / 2)
    wc2 = fc2 / (fs / 2)

    # Create FIR filters
    z1 = signal.firwin(N, wc1)  # Low-pass filter
    z2 = signal.firwin(N, [wc1, wc2], pass_zero=False)  # Band-pass filter
    z3 = signal.firwin(N, wc2, pass_zero=False)  # High-pass filter

    # Apply the filters with specified gains
    y1 = g1 * signal.lfilter(z1, 1, audio)
    y2 = g2 * signal.lfilter(z2, 1, audio)
    y3 = g3 * signal.lfilter(z3, 1, audio)

    # Combine the output and apply volume
    yT = (y1 + y2 + y3) * volume
    return yT

def load_audio(file_path):
    fs, audio = wavfile.read(file_path)
    return fs, audio

def plot_audio(signal, title):
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    st.pyplot(plt)

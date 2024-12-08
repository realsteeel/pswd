import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import streamlit as st

def generate_bintang_kecil(fs):
    freqs = {'c1': 523.3, 'd1': 587.3, 'e1': 659.3, 'f1': 698.5, 'g1': 392.0, 'g3': 785.0, 'a': 440.0, 'b': 493.9}
    duration_short = 0.70
    duration_long = 1.0
    t1 = np.arange(0, duration_short, 1/fs)
    t5 = np.arange(0, duration_long, 1/fs)
    
    c1 = np.sin(2 * np.pi * freqs['c1'] * t1)
    c2 = np.sin(2 * np.pi * freqs['c1'] * t5)
    d1 = np.sin(2 * np.pi * freqs['d1'] * t1)
    g1 = np.sin(2 * np.pi * freqs['g1'] * t1)
    b = np.sin(2 * np.pi * freqs['b'] * t1)
    nol = np.zeros(len(t1))
    
    lagu1 = np.concatenate([g1, d1, c2, nol, b, g1, nol])
    bintangkecil = np.concatenate([lagu1])
    
    sd.play(bintangkecil, fs)
    sd.wait()
    wavfile.write('hasil\Hasil.wav', fs, bintangkecil.astype(np.float32))
    st.write("Audio berhasil dimainkan dan disimpan sebagai 'Hasil.wav'.")

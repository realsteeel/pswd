import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import freqz

def plot_freq_response(hn):
    w, h = freqz(hn)
    fig, ax = plt.subplots()
    ax.plot(w, 20 * np.log10(abs(h)))
    ax.set_title('Frequency Response')
    ax.set_xlabel('Frequency [radians / sample]')
    ax.set_ylabel('Amplitude [dB]')
    ax.grid()
    st.pyplot(fig)

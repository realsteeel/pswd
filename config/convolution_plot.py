# convolution_plot.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_convolution():
    # Input for the two signals from the user
    xn = st.text_input("Masukkan sinyal pertama x(n) (contoh: 1 -1):", "1 -1")
    hn = st.text_input("Masukkan sinyal kedua h(n) (contoh: 1 -2 3):", "1 -2 3")
    
    # Convert input strings into lists of integers
    try:
        xn = list(map(int, xn.split()))
        hn = list(map(int, hn.split()))
    except ValueError:
        st.warning("Pastikan Anda memasukkan sinyal dalam format yang benar.")
        return

    # Perform convolution
    yn = np.convolve(xn, hn)

    # Create a subplot for the signals
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    # Plot h(n)
    ax1.stem(hn)
    ax1.set_title('h(n)')
    ax1.grid(True)

    # Plot x(n)
    ax2.stem(xn)
    ax2.set_title('x(n)')
    ax2.grid(True)

    # Plot the convolution result
    ax3.stem(yn)
    ax3.set_title('h(n) * x(n)')
    ax3.grid(True)

    # Adjust layout and display the plots
    plt.tight_layout()
    st.pyplot(fig)

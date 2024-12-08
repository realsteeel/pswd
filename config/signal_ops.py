# signal_processing/signal_ops.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def sum_signals(x1, x2):
    n = np.arange(len(x1))
    x_sum = x1 + x2
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create 1 row and 3 columns of subplots
    
    ax[0].stem(n, x1)
    ax[0].set_title('Sinyal x1(n)')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('x1(n)')
    
    ax[1].stem(n, x2)
    ax[1].set_title('Sinyal x2(n)')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('x2(n)')
    
    ax[2].stem(n, x_sum)
    ax[2].set_title('Hasil Penjumlahan x1(n) + x2(n)')
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('x1(n) + x2(n)')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

def time_reversal(x):
    y = np.flip(x)
    n = np.arange(len(x))
    n_reversed = -np.flip(n)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # Create 2 rows and 1 column of subplots
    
    ax[0].stem(n, x)
    ax[0].set_title('Sinyal Asli x(n)')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('x(n)')
    ax[0].grid()
    
    ax[1].stem(n_reversed, y)
    ax[1].set_title('Sinyal Hasil Pencerminan y(n) = x(-n)')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('y(n)')
    ax[1].grid()
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

def multiply_signals(x1, x2):
    n = np.arange(len(x1))
    x_product = x1 * x2
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create 1 row and 3 columns of subplots
    
    ax[0].stem(n, x1)
    ax[0].set_title('Sinyal x1(n)')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('x1(n)')
    
    ax[1].stem(n, x2)
    ax[1].set_title('Sinyal x2(n)')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('x2(n)')
    
    ax[2].stem(n, x_product)
    ax[2].set_title('Hasil Perkalian x1(n) . x2(n)')
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('x1(n) * x2(n)')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

def subtract_signals(x1, x2):
    n1 = np.arange(len(x1))
    n2 = np.arange(len(x2))
    x_sub = x1 - x2
    
    fig, ax = plt.subplots(3, 1, figsize=(15, 5))  # Create 3 rows and 1 column of subplots
    
    ax[0].stem(n1, x1)
    ax[0].set_title('Sinyal x1(n)')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('x1(n)')
    
    ax[1].stem(n2, x2)
    ax[1].set_title('Sinyal x2(n)')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('x2(n)')
    
    ax[2].stem(n1, x_sub)
    ax[2].set_title('Hasil Pengurangan x1(n) - x2(n)')
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('x1(n) - x2(n)')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

def amplitude_scaling(x, scaling_factor):
    y_amplitude_scaled = scaling_factor * x
    n = np.arange(len(x))
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # Create 2 rows and 1 column of subplots
    
    ax[0].stem(n, x)
    ax[0].set_title('Sinyal Asli')
    ax[0].set_xlabel('Indeks Sampel')
    ax[0].set_ylabel('Amplitudo')
    
    ax[1].stem(n, y_amplitude_scaled)
    ax[1].set_title('Sinyal dengan Amplitude Scaling')
    ax[1].set_xlabel('Indeks Sampel')
    ax[1].set_ylabel('Amplitudo')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

def time_shift(x, k):
    if k >= 0:
        y = np.pad(x[:-k], (k, 0), 'constant', constant_values=(0, 0))
    else:
        y = np.pad(x[-k:], (0, -k), 'constant', constant_values=(0, 0))

    n = np.arange(len(x))
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # Create 2 rows and 1 column of subplots
    
    ax[0].stem(n, x)
    ax[0].set_title('x(n)')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('x(n)')
    
    ax[1].stem(n, y)
    ax[1].set_title(f'y(n) = x(n - {k})' if k >= 0 else f'y(n) = x(n + {abs(k)})')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('y(n)')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

def time_scaling(x, scaling_factor):
    n = np.arange(len(x))
    n_scaled = n * (1 / scaling_factor)
    n_interp = np.arange(min(n_scaled), max(n_scaled) + 1)
    x_scaled = np.interp(n_interp, n_scaled, x, left=0, right=0)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # Create 2 rows and 1 column of subplots
    
    ax[0].stem(n, x)
    ax[0].set_title('Original Discrete Signal')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('Amplitude')
    
    ax[1].stem(n_interp, x_scaled)
    ax[1].set_title(f'Time-Scale Sinyal Diskrit (Scaling Factor: {scaling_factor})')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('Amplitude')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the figure to Streamlit

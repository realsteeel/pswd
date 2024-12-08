import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal

def zplane_plot(hn):
    b, a = hn, 1
    z, p, k = signal.tf2zpk(b, [a])
    
    fig, ax = plt.subplots()

    # Plot unit circle
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(unit_circle)

    # Plot zeros and poles
    ax.scatter(np.real(z), np.imag(z), color='red', marker='x', label='Zeros')  # Zero markers changed to 'x'
    ax.scatter(np.real(p), np.imag(p), color='blue', marker='o', label='Poles') # Poles markers remain 'o'

    # Set labels and title
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Z-Plane')
    
    # Configure axis
    ax.grid(True)
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    ax.set_aspect('equal')  # Ensure the plot is square
    
    # Set axis limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    
    ax.legend()
    st.pyplot(fig)

import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/signal_operations', methods=['POST'])
def signal_operations():
    data = request.json
    operation = data['operation']
    x1 = np.array(data['x1'])
    x2 = np.array(data['x2'])

    if operation == 'add':
        result = x1 + x2
    elif operation == 'subtract':
        result = x1 - x2
    elif operation == 'multiply':
        result = x1 * x2
    elif operation == 'convolve':
        result = np.convolve(x1, x2)

    return jsonify({'result': result.tolist()})

from scipy.fft import fft

@app.route('/fourier_transform', methods=['POST'])
def fourier_transform():
    data = request.json
    signal = np.array(data['signal'])
    fs = data['fs']

    # FFT computation
    F = np.abs(fft(signal))
    F = F[:len(F)//2]  # Take positive frequencies
    F = F / max(F)  # Normalize amplitude
    freq = np.linspace(0, fs/2, len(F))

    return jsonify({'frequency': freq.tolist(), 'magnitude': F.tolist()})
 
import soundfile as sf

@app.route('/audio_process', methods=['POST'])
def audio_process():
    file = request.files['file']
    fs = int(request.form['fs'])
    data, samplerate = sf.read(file)

    # Apply FFT
    F = np.abs(fft(data))
    F = F[:len(F)//2] / max(F)
    freq = np.linspace(0, samplerate/2, len(F))

    return jsonify({'frequency': freq.tolist(), 'magnitude': F.tolist()})

@app.route('/ecg_process', methods=['POST'])
def ecg_process():
    data = request.json
    ecg_signal = np.array(data['ecg'])
    fs = data['fs']
    amplitude_scale = data['scale']

    # Normalize ECG
    ecg_signal = ecg_signal / amplitude_scale
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    t = np.linspace(0, len(ecg_signal)/fs, len(ecg_signal))

    return jsonify({'time': t.tolist(), 'ecg': ecg_signal.tolist()})

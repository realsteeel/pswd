import streamlit as st
import numpy as np
from config.zplane import zplane_plot
from config.ztransform import z_transform_examples
from config.audio import generate_bintang_kecil
from config.freqresponse import plot_freq_response
from config.fft_audio import plot_fft
from config.ecg_processing import process_ecg
from config.convolution_plot import plot_convolution
from config.image_filters import high_pass_filter, low_pass_filter, upload_image
from config.audio_equalizer import equalizer, load_audio, plot_audio
from config.signal_ops import sum_signals, time_reversal, multiply_signals, subtract_signals, amplitude_scaling, time_shift, time_scaling

st.title("Sinyal dan Sistem Interaktif")

# Main menu
main_menu = ["Transformasi Z dan Fourier", "Sistem Diskret", "Filter",  "ADC, DAC & SIGNAL"]
main_choice = st.sidebar.selectbox("Pilih Kategori Program:", main_menu)

if main_choice == "Filter":
    filter_choice = st.sidebar.selectbox("Pilih Filter:", ["High Pass Filter", "Low Pass Filter", "Audio Equalizer"])

    if filter_choice == "High Pass Filter":
        st.subheader("High Pass Filter")
        image = upload_image()
        if image is not None:
            kernel = st.text_input("Masukkan kernel 3x3 (contoh: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]):", '[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]')
            kernel = np.array(eval(kernel))
            filtered_image = high_pass_filter(image, kernel)
            st.image(filtered_image, caption="Filtered Image", channels="GRAY")

    elif filter_choice == "Low Pass Filter":
        st.subheader("Low Pass Filter")
        image = upload_image()
        if image is not None:
            kernel = st.text_input("Masukkan kernel (contoh: [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]):", '[[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]')
            kernel = np.array(eval(kernel))
            filtered_image = low_pass_filter(image, kernel)
            st.image(filtered_image, caption="Filtered Image", channels="GRAY")

    elif filter_choice == "Audio Equalizer":
        st.subheader("Audio Equalizer")
        audio_file = st.file_uploader("Upload Audio File (.wav)", type=['wav'])
        if audio_file is not None:
            fs, audio = load_audio(audio_file)
            g1 = st.slider("Gain for Low Frequency:", 0.0, 1.0, 0.5)
            g2 = st.slider("Gain for Band Frequency:", 0.0, 1.0, 0.5)
            g3 = st.slider("Gain for High Frequency:", 0.0, 1.0, 0.5)
            volume = st.slider("Volume:", 0.0, 1.0, 0.5)
            if st.button("Apply Equalizer"):
                output_audio = equalizer(audio, fs, g1, g2, g3, volume)
                plot_audio(output_audio, title='Equalized Audio Output')

# Submenu untuk "Transformasi Z dan Fourier"
if main_choice == "Transformasi Z dan Fourier":
    sub_menu = ["Z-Plane Plot", "Z-Transform", "Generate Audio", "Frequency Response", "FFT Audio File", "ECG Signal Processing"]
    choice = st.sidebar.selectbox("Pilih Program:", sub_menu)

    if choice == "Z-Plane Plot":
        st.subheader("Z-Plane Plot")
        hn = st.text_input("Masukkan sinyal (contoh: [0.5, 1, -0.5, -1]):", "[0.5, 1, -0.5, -1]")
        hn = list(map(float, hn.strip('[]').split(',')))
        if st.button("Plot Z-Plane"):
            zplane_plot(hn)

    elif choice == "Z-Transform":
        st.subheader("Z-Transform")
        if st.button("Tampilkan Contoh Z-Transform"):
            z_transform_examples()

    elif choice == "Generate Audio":
        st.subheader("Generate Lagu Audio")
        fs = st.number_input("Frekuensi Sampling (Hz):", 44100)
        if st.button("Generate Lagu"):
            generate_bintang_kecil(fs)

    elif choice == "Frequency Response":
        st.subheader("Frequency Response")
        hn = st.text_input("Masukkan bentuk sinyal (contoh: [0.5, 1, -0.5, -1]):", "[0.5, 1, -0.5, -1]")
        hn = list(map(float, hn.strip('[]').split(',')))
        if st.button("Plot Frequency Response"):
            plot_freq_response(hn)

    elif choice == "FFT Audio File":
        st.subheader("FFT of Audio File - Audio")
        if st.button("Plot FFT"):
            plot_fft()

    elif choice == "ECG Signal Processing":
        st.subheader("ECG Signal Processing")
        process_ecg()

# Submenu untuk "Sistem Diskret"
elif main_choice == "Sistem Diskret":
    st.subheader("Sistem Diskret")
    st.write("Pilihan ini berfokus pada sistem diskret, seperti konvolusi.")

    # Memanggil fungsi plot_convolution yang sudah dibuat
    plot_convolution()
    
    
# Submenu for "ADC, DAC & SIGNAL"
if main_choice == "ADC, DAC & SIGNAL":
    signal_menu = st.sidebar.selectbox("Pilih Operasi Sinyal:", ["Penjumlahan", "Pengurangan", "Perkalian", "Pencerminan", "Amplitude Scaling", "Pergeseran Waktu", "Penskalaan Waktu"])
    
    if signal_menu == "Penjumlahan":
        x1_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x1, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        x2_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x2, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        if st.button("Hitung Penjumlahan"):
            x1 = np.array(list(map(float, x1_input.split())))
            x2 = np.array(list(map(float, x2_input.split())))
            sum_signals(x1, x2)

    elif signal_menu == "Pengurangan":
        x1_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x1, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        x2_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x2, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        if st.button("Hitung Pengurangan"):
            x1 = np.array(list(map(float, x1_input.split())))
            x2 = np.array(list(map(float, x2_input.split())))
            subtract_signals(x1, x2)

    elif signal_menu == "Perkalian":
        x1_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x1, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        x2_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x2, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        if st.button("Hitung Perkalian"):
            x1 = np.array(list(map(float, x1_input.split())))
            x2 = np.array(list(map(float, x2_input.split())))
            multiply_signals(x1, x2)

    elif signal_menu == "Pencerminan":
        x_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        if st.button("Hitung Pencerminan"):
            x = np.array(list(map(float, x_input.split())))
            time_reversal(x)

    elif signal_menu == "Amplitude Scaling":
        x_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        scaling_factor = st.number_input("Masukkan faktor skala amplitudo:", value=1.0)
        if st.button("Hitung Scaling"):
            x = np.array(list(map(float, x_input.split())))
            amplitude_scaling(x, scaling_factor)

    elif signal_menu == "Pergeseran Waktu":
        x_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        k = st.number_input("Masukkan nilai pergeseran waktu (k):", value=0)
        if st.button("Hitung Pergeseran"):
            x = np.array(list(map(float, x_input.split())))
            time_shift(x, k)

    elif signal_menu == "Penskalaan Waktu":
        x_input = st.text_input("Masukkan nilai-nilai sinyal diskrit x, dipisahkan dengan spasi (misalnya -1 0 1 2 3):")
        scaling_factor = st.number_input("Masukkan faktor skala waktu:", value=1.0)
        if st.button("Hitung Scaling"):
            x = np.array(list(map(float, x_input.split())))
            time_scaling(x, scaling_factor)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 29 9:10 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/smd_gain_calibration

@author: Dylan Neff, Dyn04
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    base_path = 'C:/Users/Dylan/Research/smd_calibration/'
    run_dirs = ['20240229-pulsar']
    process_data(base_path, run_dirs)
    plt.show()
    print('donzo')


def process_data(base_path, run_dirs):
    for run_dir in run_dirs:
        process_run(base_path + run_dir)


def process_run(run_dir, n_chan=4, amplitude_out_file='amplitudes.txt'):
    """

    :param run_dir:
    :param n_chan:
    :param amplitude_out_file:
    :return:
    """
    channel_amplitudes = []
    for waveform_file in os.listdir(run_dir):
        print(waveform_file)
        if waveform_file.endswith('.txt') and waveform_file != amplitude_out_file:
            channel_amplitudes.append([])
            waveform_data = np.genfromtxt(f'{run_dir}/{waveform_file}', delimiter='\t', skip_header=3)
            # print(waveform_data)
            wavform_transpose = waveform_data.T
            # time = wavform_transpose[0]
            waveform_channels = wavform_transpose[1:]
            for i in range(n_chan):
                # amplitude = analyze_waveform(waveform_channels[i], plot=i==0)
                amplitude = analyze_waveform(waveform_channels[i])
                channel_amplitudes[-1].append(amplitude)

    channel_amplitudes = np.array(channel_amplitudes).T
    plot_amplitude_distribution(channel_amplitudes, n_chan=1)
    write_amplitudes(channel_amplitudes, run_dir, amplitude_out_file)


def analyze_waveform(waveform, signal_window_frac=0.3, plot=False):
    """
    Analyze waveform data
    :param waveform:
    :param signal_window_frac: float, fraction of waveform to use for signal window
    :param plot: bool, if True plot waveform and analysis
    :return:
    """
    min_val = np.min(waveform)
    min_val_index = np.argmin(waveform)
    signal_window_size = int(len(waveform) * signal_window_frac)
    signal_window = [min_val_index - signal_window_size // 2, min_val_index + signal_window_size // 2]
    signal_window = [max(signal_window[0], 0), min(signal_window[1], len(waveform))]
    # signal_data = waveform[signal_window[0]:signal_window[1]]
    baseline_data = np.concatenate([waveform[:signal_window[0]], waveform[signal_window[1]:]])

    baseline = np.mean(baseline_data)
    signal_amplitude = min_val - baseline

    if plot:
        fig, ax = plt.subplots()
        ax.plot(waveform, label='Waveform')
        ax.axhline(min_val, color='r', linestyle='--', label='Minimum')
        ax.axhline(baseline, color='gray', linestyle='--', label='Baseline')
        ax.axvline(signal_window[0], color='b', linestyle='--', label='Signal Window')
        ax.axvline(signal_window[1], color='b', linestyle='--')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Voltage (V)')
        ax.grid()
        ax.legend()
        plt.show()

    return signal_amplitude


def write_amplitudes(channel_amplitudes, run_dir, amplitude_out_file):
    with open(f'{run_dir}/{amplitude_out_file}', 'w') as file:
        header_str = '\t'.join([f'Channel {i}' for i in range(len(channel_amplitudes))]) + '\n'
        file.write(header_str)
        for i in range(len(channel_amplitudes[0])):
            trigger_str = '\t'.join([f'{channel_amplitudes[j][i]}' for j in range(len(channel_amplitudes))]) + '\n'
            file.write(trigger_str)


def read_amplitudes(run_dir, amplitude_out_file):
    channel_amplitudes = np.genfromtxt(f'{run_dir}/{amplitude_out_file}', delimiter='\t', skip_header=1)
    return channel_amplitudes


def plot_waveforms(wavform_data, n_chan=4):
    fig, ax = plt.subplots()
    for i in range(n_chan):
        ax.plot(wavform_data[0], wavform_data[i + 1], label=f'Channel {i}')
    ax.grid()
    ax.legend()


def plot_amplitude_distribution(channel_amplitudes, n_chan=4):
    fig, ax = plt.subplots()
    for i in range(n_chan):
        ax.hist(channel_amplitudes[i], bins=20, label=f'Channel {i}', alpha=0.5)
        ax.axvline(np.mean(channel_amplitudes[i]), color='r', linestyle='--', label=f'Channel {i} Mean')
    ax.set_xlabel('Signal Amplitude (V)')
    ax.set_xlim(-0.8, 0.1)
    ax.grid()
    ax.legend()
    fig.tight_layout()


if __name__ == '__main__':
    main()

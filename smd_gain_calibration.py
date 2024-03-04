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
from scipy.optimize import curve_fit as cf


def main():
    base_path = 'C:/Users/Dylan/Research/smd_calibration/'
    # base_path = '/local/home/dn277127/Documents/smd_calibration/'
    # run_dirs = ['20240301-chA', '20240301-chB']
    run_dirs = ['20240301-pp1_h1', '20240301-pp2_h2', '20240301-pp3_h3', '20240301-pp4_h4', '20240301-pp5_h5']
    # run_dirs = os.listdir(base_path)
    channels = [0]
    reprocess = False
    process_data(base_path, run_dirs, channels, reprocess)
    # convert_to_csv(base_path, run_dirs)
    # analyze_baselines(base_path, run_dirs, channels)
    plt.show()
    print('donzo')


def process_data(base_path, run_dirs, channels, reprocess=False):
    for run_dir in run_dirs:
        process_run(base_path + run_dir, channels, reprocess=reprocess)


def analyze_baselines(base_path, run_dirs, channels=None):
    """
    Extract just baselines from all runs and plot distribution
    :param base_path:
    :param run_dirs:
    :param channels:
    :return:
    """
    amplitude_out_file = 'amplitudes.txt'
    if channels is None:
        channels = [0, 1, 2, 3]
    baselines = {chan_i: [] for chan_i in channels}
    for run_dir in run_dirs:
        for waveform_file in os.listdir(base_path + run_dir):
            if waveform_file.endswith('.csv') and waveform_file != amplitude_out_file:
                print(f'Processing {waveform_file}')
                waveform_data = np.genfromtxt(f'{base_path}{run_dir}/{waveform_file}', delimiter=',', skip_header=3)
                wavform_transpose = waveform_data.T
                waveform_channels = wavform_transpose[1:]
                for chan_i in channels:
                    base, sw, baseline_data = get_baseline(waveform_channels[chan_i], return_baseline_data=True)
                    baselines[chan_i].extend(baseline_data)
    plot_baseline_distribution(baselines, title=f'Baseline Distribution')


def process_run(run_dir, channels=None, amplitude_out_file='amplitudes.txt', reprocess=False):
    """

    :param run_dir:
    :param channels:
    :param amplitude_out_file:
    :param reprocess:
    :return:
    """
    if channels is None:
        channels = [0, 1, 2, 3]
    if not reprocess and os.path.exists(f'{run_dir}/{amplitude_out_file}'):
        channel_amplitudes = read_amplitudes(run_dir, amplitude_out_file)
    else:
        channel_amplitudes = []
        for waveform_file in os.listdir(run_dir):
            print(waveform_file)
            if waveform_file.endswith('.csv') and waveform_file != amplitude_out_file:
                channel_amplitudes.append([])
                waveform_data = np.genfromtxt(f'{run_dir}/{waveform_file}', delimiter=',', skip_header=3)
                wavform_transpose = waveform_data.T
                # time = wavform_transpose[0]
                waveform_channels = wavform_transpose[1:]
                for chan_i in channels:
                    # amplitude = analyze_waveform(waveform_channels[i], plot=i==0)
                    amplitude = analyze_waveform(waveform_channels[chan_i], plot=False)
                    channel_amplitudes[-1].append(amplitude)

        channel_amplitudes = np.array(channel_amplitudes).T
        write_amplitudes(channel_amplitudes, run_dir, amplitude_out_file)
    print(channel_amplitudes)
    title = run_dir.split('/')[-1]
    plot_amplitude_distribution(channel_amplitudes, channels, title)


def analyze_waveform(waveform, signal_window_frac=0.3, plot=False):
    """
    Analyze waveform data
    :param waveform:
    :param signal_window_frac: float, fraction of waveform to use for signal window
    :param plot: bool, if True plot waveform and analysis
    :return:
    """
    min_val = np.min(waveform)
    # min_val_index = np.argmin(waveform)
    # signal_window_size = int(len(waveform) * signal_window_frac)
    # signal_window = [min_val_index - signal_window_size // 2, min_val_index + signal_window_size // 2]
    # signal_window = [max(signal_window[0], 0), min(signal_window[1], len(waveform))]
    # # signal_data = waveform[signal_window[0]:signal_window[1]]
    # baseline_data = np.concatenate([waveform[:signal_window[0]], waveform[signal_window[1]:]])

    # baseline = np.mean(baseline_data)
    baseline, signal_window = get_baseline(waveform, signal_window_frac)
    signal_amplitude = min_val - baseline

    if plot:
        fig, ax = plt.subplots()
        ax.plot(waveform, label='Waveform')
        ax.axhline(min_val, color='r', linestyle='--', label='Minimum')
        ax.axhline(baseline, color='gray', linestyle='--', label='Baseline')
        ax.axvline(signal_window[0], color='b', linestyle='--', label='Signal Window')
        ax.axvline(signal_window[1], color='b', linestyle='--')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Voltage (mV)')
        ax.grid()
        ax.legend()
        plt.show()

    return signal_amplitude


def get_baseline(waveform, signal_window_frac=0.3, return_baseline_data=False):
    """
    Get baseline of waveform
    :param waveform:
    :param signal_window_frac:
    :param return_baseline_data:
    :return:
    """
    # min_val = np.min(waveform)
    min_val_index = np.argmin(waveform)
    signal_window_size = int(len(waveform) * signal_window_frac)
    signal_window = [min_val_index - signal_window_size // 2, min_val_index + signal_window_size // 2]
    signal_window = [max(signal_window[0], 0), min(signal_window[1], len(waveform))]
    baseline_data = np.concatenate([waveform[:signal_window[0]], waveform[signal_window[1]:]])
    baseline = np.mean(baseline_data)
    if return_baseline_data:
        return baseline, signal_window, baseline_data
    return baseline, signal_window


def write_amplitudes(channel_amplitudes, run_dir, amplitude_out_file):
    with open(f'{run_dir}/{amplitude_out_file}', 'w') as file:
        header_str = '\t'.join([f'Channel {i}' for i in range(len(channel_amplitudes))]) + '\n'
        file.write(header_str)
        for i in range(len(channel_amplitudes[0])):
            trigger_str = '\t'.join([f'{channel_amplitudes[j][i]}' for j in range(len(channel_amplitudes))]) + '\n'
            file.write(trigger_str)


def read_amplitudes(run_dir, amplitude_out_file):
    channel_amplitudes = np.genfromtxt(f'{run_dir}/{amplitude_out_file}', delimiter='\t', skip_header=1)
    if len(channel_amplitudes.shape) == 1:
        channel_amplitudes = [channel_amplitudes]
    return channel_amplitudes


def plot_waveforms(wavform_data, n_chan=4):
    fig, ax = plt.subplots()
    for i in range(n_chan):
        ax.plot(wavform_data[0], wavform_data[i + 1], label=f'Channel {i}')
    ax.grid()
    ax.legend()


def plot_amplitude_distribution(channel_amplitudes, channels=None, title=None, fit=False):
    if channels is None:
        channels = [0, 1, 2, 3]
    fig, ax = plt.subplots()
    for chan_i in channels:
        # ax.hist(channel_amplitudes[chan_i], bins=20, label=f'Channel {chan_i}', alpha=0.5)
        amps = -channel_amplitudes[chan_i]  # Make positive
        hist, bins = np.histogram(amps, bins=20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, hist, width=(bins[1] - bins[0]), label=f'Channel {chan_i}', alpha=0.5, align='center')
        # ax.axvline(np.mean(channel_amplitudes[chan_i]), color='r', linestyle='--', label=f'Channel {chan_i} Mean')
        if fit:
            p0 = [np.max(hist), 7.5, np.std(amps)]
            bounds = ([np.max(hist) * 0.1, 0, 0], [np.max(hist) * 10, 20, np.inf])
            print(p0)
            func = gaussian
            fit_cutoff = 8
            ax.axvline(fit_cutoff, color='purple', linestyle='--', label='Fit Cutoff')
            bin_centers_fit, hist_fit = bin_centers[bin_centers > fit_cutoff], hist[bin_centers > fit_cutoff]
            hist_fit_errs = np.where(hist_fit == 0, 1, np.sqrt(hist_fit))
            x_plot = np.linspace(0, max(bin_centers), 1000)
            ax.plot(x_plot, func(x_plot, *p0), color='gray', label=f'Channel {chan_i} Guess')
            popt, pcov = cf(func, bin_centers_fit, hist_fit, sigma=hist_fit_errs, p0=p0, bounds=bounds,
                            absolute_sigma=True)
            ax.plot(x_plot, func(x_plot, *popt), label=f'Channel {chan_i} Fit')
    ax.set_xlabel('Signal Amplitude (V)')
    if title is not None:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    fig.tight_layout()


def plot_baseline_distribution(baselines, title=None):
    fig, ax = plt.subplots()
    for chan_i, baseline_data in baselines.items():
        ax.hist(baseline_data, bins=50, alpha=0.5, label=f'Channel {chan_i}')
    ax.set_xlabel('Baseline (mV)')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.grid()
    fig.tight_layout()


def convert_to_csv(base_path, run_dirs):
    for run_dir in run_dirs:
        print(f'\nConverting {run_dir}:')
        for waveform_file in os.listdir(base_path + run_dir):
            if waveform_file.endswith('.txt') and waveform_file != 'amplitudes.txt':
                print(f'Converting {waveform_file}')
                with open(f'{base_path}{run_dir}/{waveform_file}', 'r') as file:
                    text = file.read()
                text = text.replace('\t', ',')
                with open(f'{base_path}{run_dir}/{waveform_file[:-4]}.csv', 'w') as file:
                    file.write(text)


def gaussian(x, a, mu, sig):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# def landau(x, a, mu, sig):
#     return a * np.exp(-(x - mu) / sig) * (1 + (x - mu) / sig)


if __name__ == '__main__':
    main()

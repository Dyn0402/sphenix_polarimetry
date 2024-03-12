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
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.optimize import curve_fit as cf
import pylandau
from multiprocessing import Pool
import tqdm
import istarmap  # This is a custom module that adds starmap to multiprocessing.pool

from Measure import Measure


def main():
    side = 'North'
    base_path = f'C:/Users/Dylan/Research/smd_calibration/march6_{side}SMD/'
    fig_save_dir = 'C:/Users/Dylan/Research/smd_calibration/plots/'
    pdf_save_path = f'{fig_save_dir}{side}_smd_calibration.pdf'
    south_let = 's-' if side == 'South' else ''
    # base_path = '/local/home/dn277127/Documents/smd_calibration/march6_SouthSMD/'
    # fig_save_dir = '/local/home/dn277127/Documents/smd_calibration/plots/'
    # run_dirs = ['20240301-chA', '20240301-chB']
    # run_dirs = ['20240301-pp1_h1', '20240301-pp2_h2', '20240301-pp3_h3', '20240301-pp4_h4', '20240301-pp5_h5']
    run_dirs = ['20240306-s-ch13']
    bkg_dirs = ['20240306-s-bg-ch13']
    sig_dirs = ['20240306-s-ch13', '20240306-s-ch13-wsource-nolead']
    # run_dirs = os.listdir(base_path)
    channels = [0]
    reprocess = True
    threads = 15
    # process_data(base_path, run_dirs, channels, reprocess, threads, fig_save_dir)
    if pdf_save_path is not None:
        pdf = PdfPages(pdf_save_path)
    else:
        pdf = None
    smd_channels, gaus_centers = [], []
    for i in range(1, 16):
        bkg_dir = f'20240306-{south_let}bg-ch{i}'
        if os.path.exists(f'{base_path}{bkg_dir}new'):
            bkg_dir = f'{bkg_dir}new'
        sig_dir = f'20240306-{south_let}ch{i}'
        if os.path.exists(f'{base_path}{sig_dir}new'):
            sig_dir = f'{sig_dir}new'
        run_dirs = [bkg_dir] + [sig_dir]
        # process_data(base_path, run_dirs, channels, reprocess, threads, fig_save_dir)
        gaus_center = data_analaysis(base_path, sig_dir, bkg_dir, channels, fig_save_dir, pdf)[0]
        smd_channels.append(i)
        gaus_centers.append(gaus_center)
        # analyze_pulse_shapes(base_path, run_dirs, channels, fig_save_dir)
    plot_peaks(smd_channels, gaus_centers, fig_save_dir)
    # convert_to_csv(base_path, run_dirs)
    # analyze_baselines(base_path, run_dirs, channels)
    # analyze_pulse_shapes(base_path, run_dirs, channels)
    pdf.close()
    plt.show()
    print('donzo')


def process_data(base_path, run_dirs, channels, reprocess=False, threads=8, fig_save_dir=None):
    for run_dir in run_dirs:
        process_run(base_path + run_dir, channels, reprocess=reprocess, threads=threads, fig_save_dir=fig_save_dir)


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


def analyze_pulse_shapes(base_path, run_dirs, channels=None, fig_save_dir=None):
    """

    :param base_path:
    :param run_dirs:
    :param channels:
    :param fig_save_dir:
    :return:
    """
    threads = 8
    max_amp = 105
    amplitude_out_file = 'amplitudes.txt'
    if channels is None:
        channels = [0, 1, 2, 3]
    for run_dir in run_dirs:
        run_dir = base_path + run_dir
        channel_amplitudes, channel_sums = [], []
        with Pool(threads) as pool:
            jobs = [(file, run_dir, amplitude_out_file, channels, max_amp, True) for file in os.listdir(run_dir)
                    if file.endswith('.csv') and file != amplitude_out_file]
            for channel_amps, chan_sums in tqdm.tqdm(pool.istarmap(process_waveform, jobs), total=len(jobs)):
                channel_amplitudes.append(channel_amps)
                channel_sums.append(chan_sums)
        channel_amplitudes = np.array(channel_amplitudes).T
        channel_sums = np.array(channel_sums).T
        print(f'Channel Amplitudes: {channel_amplitudes}')
        print(f'Channel Sums: {channel_sums}')
        plot_amp_sums(channel_amplitudes, channel_sums, channels, run_dir.split('/')[-1], fig_save_dir)


def process_run(run_dir, channels=None, amplitude_out_file='amplitudes.txt', reprocess=False, threads=8,
                fig_save_dir=None):
    """
    :param run_dir:
    :param channels:
    :param amplitude_out_file:
    :param reprocess:
    :param threads:
    :param fig_save_dir:
    :return:
    """
    max_amp = 120
    print(f'Processing {run_dir}...')
    if channels is None:
        channels = [0, 1, 2, 3]

    if not reprocess and os.path.exists(f'{run_dir}/{amplitude_out_file}'):
        channel_amplitudes = read_amplitudes(run_dir, amplitude_out_file)
    else:
        channel_amplitudes = []
        with Pool(threads) as pool:
            jobs = [(file, run_dir, amplitude_out_file, channels, max_amp) for file in os.listdir(run_dir)
                    if file.endswith('.csv') and file != amplitude_out_file]
            for channel_amps in tqdm.tqdm(pool.istarmap(process_waveform, jobs), total=len(jobs)):
                channel_amplitudes.append(channel_amps)
        channel_amplitudes = np.array(channel_amplitudes).T
        write_amplitudes(channel_amplitudes, run_dir, amplitude_out_file)

    # print(channel_amplitudes)
    title = run_dir.split('/')[-1]
    plot_amplitude_distribution(channel_amplitudes, channels, title, False, fig_save_dir)


def data_analaysis(base_path, run_dir, bkg_dir, channels=None, fig_save_dir=None, pdf=None):
    """

    :param base_path:
    :param run_dir:
    :param bkg_dir:
    :param channels:
    :param fig_save_dir:
    :param pdf:
    :return:
    """
    max_amplitude = 110
    amplitude_out_file = 'amplitudes.txt'
    if channels is None:
        channels = [0, 1, 2, 3]
    channel_amplitudes = read_amplitudes(base_path + run_dir, amplitude_out_file)
    bkg_amplitudes = read_amplitudes(base_path + bkg_dir, amplitude_out_file)
    gaus_centers = {}
    for chan_i in channels:
        sig_amps = -np.array(channel_amplitudes[chan_i])
        sig_amps = sig_amps[sig_amps <= max_amplitude]
        bkg_amps = -np.array(bkg_amplitudes[chan_i])
        bkg_amps = bkg_amps[bkg_amps <= max_amplitude]
        min_edge = min(min(sig_amps), min(bkg_amps))
        max_edge = max(max(sig_amps), max(bkg_amps))
        bin_edges = np.linspace(min_edge, max_edge, 20)
        bkg_hist, bkg_bin_edges = np.histogram(bkg_amps, bins=bin_edges)
        sig_hist, sig_bin_edges = np.histogram(sig_amps, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        corrected_sig_hist = sig_hist - bkg_hist
        fig, ax = plt.subplots(figsize=(6.66, 5.2), dpi=144)
        ax.step(bin_centers, corrected_sig_hist, where='mid', label='Corrected Signal', color='red', lw=3)
        ax.bar(bin_centers, sig_hist, width=(bin_edges[1] - bin_edges[0]), label='Signal', alpha=0.5,
               align='center', color='blue')
        ax.bar(bin_centers, bkg_hist, width=(bin_edges[1] - bin_edges[0]), label='Background', alpha=0.5,
               align='center', color='gray')
        ax.set_xlabel('Signal Amplitude (mV)')
        ax.set_ylabel('Counts')
        ax.set_title(f'{run_dir}')
        ax.grid()
        ax.legend()
        fig.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(6.66, 5.2), dpi=144)
        ax2.bar(bin_centers, sig_hist, width=(bin_edges[1] - bin_edges[0]), label='Cs-137 Spectrum',
                alpha=0.5, align='center', color='blue')

        def langau_plus_bkg_spectrum(x, mu, eta, sig, a, a_landau, bkg_frac):
            x_bkg = find_nearest_float(bin_centers, x)
            bkg_hists = np.array([bkg_hist[x_i] for x_i in x_bkg])
            return pylandau.langau(x, mu, eta, sig, a, a_landau) + bkg_frac * bkg_hists
            # return pylandau.langau(x, mu, eta, sig, a, a_landau)

        weights = np.where(sig_hist > 0, sig_hist, 0)
        # func = gaussian
        # mean_par_num = 1
        # p0 = [np.max(sig_hist), np.average(bin_centers, weights=weights), 20]
        mean = np.average(bin_centers, weights=weights)
        lower_fit_bound = mean * 0.8
        fit_sig_amps = sig_amps[sig_amps > lower_fit_bound]
        fit_sig_hist, fit_sig_bin_edges = np.histogram(fit_sig_amps, bins=bin_edges)
        fit_bin_centers = (fit_sig_bin_edges[:-1] + fit_sig_bin_edges[1:]) / 2
        fit_sig_errs = np.where(fit_sig_hist == 0, 1, np.sqrt(fit_sig_hist))
        # func = pylandau.landau
        # mean_par_num = 0
        # func_bounds = ([0, 1, 0], [100, 10000, np.max(fit_sig_hist) * 10])
        # p0 = [mean, 2, np.max(fit_sig_hist)]
        # func = pylandau.langau
        # mean_par_num = 0
        # func_bounds = ([0, 1, 0, 0, 0], [100, 10000, 100, np.max(fit_sig_hist) * 10, 100])
        # p0 = [mean, 2, 5, np.max(fit_sig_hist), 1]
        func = langau_plus_bkg_spectrum
        mean_par_num = 0
        func_bounds = ([0, 1, 0, 0, 0, 0], [100, 10000, 100, np.max(fit_sig_hist) * 10, 100, 0.5])
        p_names = ['mu', 'eta', 'sig', 'a', 'a_landau', 'bkg_frac']
        p0 = [mean, 2, 5, np.max(fit_sig_hist), 1, 0.02]
        popt, pcov = cf(func, fit_bin_centers, fit_sig_hist, sigma=fit_sig_errs, p0=p0, bounds=func_bounds,
                        absolute_sigma=True)
        x_plot = np.linspace(min_edge, max_edge, 1000)
        ax2.plot(x_plot, func(x_plot, *p0), label='Guess', color='Gray')
        ax2.plot(bin_centers, popt[-1] * bkg_hist, label='Bkg Contribution', color='green', alpha=0.5)
        ax2.plot(x_plot, func(x_plot, *popt), label='Fit', color='red')
        ax2.axvline(lower_fit_bound, ls='--', zorder=0, color='purple', label='Fit Lower Bound')
        fit_parameter_mesures = [Measure(popt[i], np.sqrt(pcov[i][i])) for i in range(len(popt))]
        fit_parameters_string = '\n'.join([f'{p_names[i]}: {fit_parameter_mesures[i]}' for i in range(len(popt))])
        ax2.annotate(fit_parameters_string, (0.02, 0.9), xycoords='axes fraction', ha='left', va='top',
                     bbox=dict(facecolor='wheat', alpha=0.5))
        ax2.set_xlabel('Signal Amplitude (mV)')
        ax2.set_ylabel('Counts')
        ax2.set_title(f'Signal Spectrum Fit: {run_dir}')
        ax2.grid()
        ax2.legend(loc='upper right')
        fig2.tight_layout()

        if fig_save_dir is not None:
            fig.savefig(f'{fig_save_dir}{run_dir}_spectrum.png')
            fig2.savefig(f'{fig_save_dir}{run_dir}_spectrum_fit.png')

        if pdf is not None:
            pdf.savefig(fig2)

        perr = np.sqrt(np.diag(pcov))
        gaus_centers.update({chan_i: Measure(popt[mean_par_num], perr[mean_par_num])})

    return gaus_centers


def process_waveform(waveform_file, run_dir, amplitude_out_file, channels=None, max_amp=55, return_sums=False):
    if waveform_file.endswith('.csv') and waveform_file != amplitude_out_file:
        channel_amplitudes = []
        waveform_data = np.genfromtxt(f'{run_dir}/{waveform_file}', delimiter=',', skip_header=3)
        waveform_data = np.where(np.isinf(waveform_data), np.where(waveform_data < 0, -max_amp, max_amp), waveform_data)
        waveform_transpose = waveform_data.T
        waveform_channels = waveform_transpose[1:]
        for chan_i in channels:
            amplitude = analyze_waveform(waveform_channels[chan_i], plot=False)
            if np.isinf(amplitude):
                print(f'Inf amplitude in {waveform_file}')
            channel_amplitudes.append(amplitude)
        if return_sums:
            channel_sums = []
            for chan_i in channels:
                amplitude, signal_sum = analyze_waveform(waveform_channels[chan_i], plot=False, return_sum=True)
                channel_sums.append(signal_sum)
            return channel_amplitudes, channel_sums
        return channel_amplitudes


def analyze_waveform(waveform, signal_window_frac=0.3, plot=False, return_sum=False):
    """
    Analyze waveform data
    :param waveform:
    :param signal_window_frac: float, fraction of waveform to use for signal window
    :param plot: bool, if True plot waveform and analysis
    :param return_sum: bool, if True return sum of waveform
    :return:
    """
    min_val = np.min(waveform)

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

    if return_sum:
        signal_sum = np.sum(waveform[signal_window[0]:signal_window[1]] - baseline)
        return signal_amplitude, signal_sum
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


def plot_amplitude_distribution(channel_amplitudes, channels=None, title=None, fit=False, fig_save_dir=None):
    n_bins = max(len(channel_amplitudes[0]) // 500, 20)
    if channels is None:
        channels = [0, 1, 2, 3]
    fig, ax = plt.subplots(figsize=(6.66, 5.2), dpi=144)
    for chan_i in channels:
        # ax.hist(channel_amplitudes[chan_i], bins=20, label=f'Channel {chan_i}', alpha=0.5)
        amps = -channel_amplitudes[chan_i]  # Make positive
        hist, bins = np.histogram(amps, bins=n_bins)
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
    ax.set_xlabel('Signal Amplitude (mV)')
    if title is not None:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6.66, 5.2), dpi=144)
    for chan_i in channels:
        amps = -channel_amplitudes[chan_i]  # Make positive
        hist, bins = np.histogram(amps, bins=20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        sns.histplot(amps, bins=n_bins, kde=True, ax=ax2, label=f'Channel {chan_i}', alpha=0.5)
        sns.rugplot(amps, ax=ax2, color='gray', alpha=0.5)
    ax2.set_xlabel('Signal Amplitude (mV)')
    if title is not None:
        ax2.set_title(title)
    fig2.tight_layout()
    if fig_save_dir is not None:
        fig2.savefig(f'{fig_save_dir}{title}_hist_kde.png')


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


def plot_amp_sums(channel_amplitudes, channel_sums, channels, title, fig_save_dir=None):
    fig, ax = plt.subplots(figsize=(6.66, 5.2), dpi=144)
    for chan_i in channels:
        ax.scatter(channel_amplitudes[chan_i], channel_sums[chan_i], alpha=0.2, label=f'Channel {chan_i}')
    ax.set_xlabel('Signal Amplitude (mV)')
    ax.set_ylabel('Signal Sum (mV*ns)')
    ax.set_title(title)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.66, 5.2), dpi=144)

    # Plot two-dimensional histogram density heatmap
    h, xedges, yedges, img = ax.hist2d(channel_amplitudes[0], channel_sums[0], bins=100, density=False, cmin=1,
                                       norm=LogNorm())

    ax.set_xlabel('Signal Amplitude (mV)')
    ax.set_ylabel('Signal Sum (mV*ns)')
    ax.set_title(title)
    fig.colorbar(img, ax=ax)
    ax.grid()
    fig.tight_layout()
    if fig_save_dir is not None:
        fig.savefig(f'{fig_save_dir}{title}_amp_sum.png')


def plot_peaks(smd_channels, peak_centers, fig_save_dir):
    fig, ax = plt.subplots(figsize=(6.66, 5.2), dpi=144)
    peak_center_vals, peak_cener_errs = [peak.val for peak in peak_centers], [peak.err for peak in peak_centers]

    ax.errorbar(smd_channels, peak_center_vals, yerr=peak_cener_errs, fmt='o', label='Gaussian Center', color='blue',
                ls='none')
    ax.set_xlabel('SMD Channel')
    ax.set_ylabel('Gaussian Center (mV)')
    ax.set_title('Gaussian Center vs SMD Channel')
    ax.axhline(0, color='black', linestyle='-')
    ax.axhline(100, color='black', linestyle='-')
    ax.axhline(np.mean(peak_center_vals), color='red', alpha=0.3, linestyle='-', label='Mean')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{fig_save_dir}gaus_center_vs_smd_chan.png')


def find_nearest_float(float_array, input_float):
    idx = np.abs(np.subtract.outer(float_array, input_float)).argmin(axis=0)
    # nearest_floats = float_array[idx]
    return idx


def gaussian(x, a, mu, sig):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':
    main()

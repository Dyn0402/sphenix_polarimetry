#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 13 10:55 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/south_comparison_greg.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    smd_channels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    dylan_peaks_south = np.array([42.888381795167696, 34.76024276141641, 39.44993874222971, 44.25536692115893, 41.062659751621894, 40.53554335572886, 51.779895509634734, 52.54943451529954, 42.63357933923833, 41.564550822299864, 50.724966191756856, 45.43376896177092, 42.26135196762757, 39.71151093636002, 45.91228258311852])
    dylan_errs_south = np.array([0.2520887182919364, 0.1793128790643829, 0.2304657359290988, 0.2637838293054689, 0.2624798662330775, 0.25129914618491667, 0.34146937555361634, 0.3715546296383848, 0.34836040303257343, 0.314492241920684, 0.3701335361386798, 0.30882126125341103, 0.26374937679272187, 0.26779849875060396, 0.3476758362927751])
    dylan_peaks_north = np.array([52.64116262719443, 44.91362342197709, 42.01146561349985, 42.63786686706889, 50.92935854791357, 47.385251343151886, 53.24781438219798, 57.09282185074957, 58.527989898178625, 47.9151780233233, 49.76175953257518, 48.1401960507262, 51.11488605746357, 53.63089610638185, 58.74039790327372])
    dylan_errs_north = np.array([0.30544100833995536, 0.2447744065364325, 0.21253628856347995, 0.21965124844420658, 0.2920583710615548, 0.7238545948477134, 0.3543395138857282, 0.8213941762954459, 0.3970796030204108, 0.3481951080147322, 0.7637621645295218, 0.2887427217484494, 0.3081605090408349, 0.3536600926941165, 0.4187890351163449])
    greg_peaks_north = np.array([55.913446, 46.289342, 41.733395, 43.044779, 52.197371, 43.811537, 54.923654, 60.980272, 61.541491, 42.293098, 44.844006, 45.915014, 50.512490, 54.933794, 67.347288])
    greg_errs_north = np.array([0 for i in range(len(smd_channels))])
    greg_peaks_south = np.array([44.691067, 37.721292, 40.889184, 46.000972, 40.443478, 37.684823, 52.197187, 54.056663, 43.427500, 40.565529, 41.781205, 40.089429, 40.595987, 39.754156, 44.838982])
    greg_errs_south = np.array([0 for i in range(len(smd_channels))])

    y_lim = [30, 70]

    fig_south, (ax_south, ax_south_diff) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [7, 3]}, sharex='all')
    ax_south.errorbar(smd_channels, dylan_peaks_south, color='green', yerr=dylan_errs_south, fmt='o', label='Dylan')
    ax_south.errorbar(smd_channels, greg_peaks_south, color='blue', yerr=greg_errs_south, fmt='o', label='Greg')
    percent_diff_s = (dylan_peaks_south - greg_peaks_south) / greg_peaks_south
    percent_diff_s_err = (dylan_errs_south**2 + greg_errs_south**2)**0.5 / greg_peaks_south
    ax_south_diff.errorbar(smd_channels, percent_diff_s, color='red', yerr=percent_diff_s_err, fmt='o', label='Dylan - Greg')
    ax_south_diff.axhline(0, color='black', linestyle='-', alpha=0.7, zorder=0)
    ax_south.set_title('South SMD Gain Calibration')
    ax_south_diff.set_xlabel('SMD Channel')
    ax_south.set_ylabel('Peak Position (mV)')
    ax_south_diff.set_ylabel('% Difference')
    ax_south.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_south.legend(loc='upper left')
    ax_south_diff.legend(loc='upper left')
    ax_south_diff.yaxis.set_major_formatter(plt.FuncFormatter(percent_formatter))
    ax_south.set_ylim(y_lim)
    fig_south.tight_layout()
    fig_south.subplots_adjust(hspace=0.01)

    fig_north, (ax_north, ax_north_diff) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [7, 3]}, sharex='all')
    ax_north.errorbar(smd_channels, dylan_peaks_north, color='green', yerr=dylan_errs_north, fmt='o', label='Dylan')
    ax_north.errorbar(smd_channels, greg_peaks_north, color='blue', yerr=greg_errs_north, fmt='o', label='Greg')
    percent_diff_n = (dylan_peaks_north - greg_peaks_north) / greg_peaks_north
    percent_diff_n_err = (dylan_errs_north**2 + greg_errs_north**2)**0.5 / greg_peaks_north
    ax_north_diff.errorbar(smd_channels, percent_diff_n, color='red', yerr=percent_diff_n_err, fmt='o', label='Dylan - Greg')
    ax_north_diff.axhline(0, color='black', linestyle='-', alpha=0.7, zorder=0)
    ax_north.set_title('North SMD Gain Calibration')
    ax_north_diff.set_xlabel('SMD Channel')
    ax_north.set_ylabel('Peak Position (mV)')
    ax_north_diff.set_ylabel('% Difference')
    ax_north.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_north.legend(loc='upper left')
    ax_north_diff.legend(loc='upper left')
    ax_north_diff.yaxis.set_major_formatter(plt.FuncFormatter(percent_formatter))
    ax_north.set_ylim(y_lim)
    fig_north.tight_layout()
    fig_north.subplots_adjust(hspace=0.01)

    plt.show()

    print('donzo')


def percent_formatter(x, pos):
    return '{:.0f}%'.format(x * 100)


if __name__ == '__main__':
    main()

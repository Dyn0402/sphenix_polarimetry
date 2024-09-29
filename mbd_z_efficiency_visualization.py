#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 26 3:46 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/mbd_z_efficiency_visualization.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    widths = [175, 500, 1000]  # cm
    amp = 0.57

    x_lim = (-219, 219)

    x = np.linspace(x_lim[0], x_lim[1], 1000)
    fig, ax = plt.subplots()
    fig_norm, ax_norm = plt.subplots()

    for width in widths:
        y = gaus(x, amp, 0, width)
        ax.plot(x, y, label=f'Width: {width} cm')
        y_norm = gaus(x, 1, 0, width)
        if width == 500:
            ax_norm.plot(x, y_norm, color='orange', label=f'Width: {width} cm')

    ax.axhline(amp, color='gray', linestyle='--', label='Amplitude')
    ax.axhline(0.52, color='gray', linestyle='--', label='Minimum Efficiency')
    ax.set_xlim(x_lim)
    ax.set_ylim(0, 1.09)
    ax.set_xlabel(r'$v_z^{truth}$')
    ax.set_ylabel('Efficiency')
    ax.legend()
    fig.tight_layout()

    ax_norm.axhline(1, color='gray', linestyle='--')
    ax_norm.set_xlim(x_lim)
    ax_norm.set_ylim(0, 1.09)
    ax_norm.set_xlabel(r'$v_z^{truth}$')
    ax_norm.set_ylabel('Relative Efficiency')
    ax_norm.legend()
    fig_norm.tight_layout()

    plt.show()

    print('donzo')


def gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


if __name__ == '__main__':
    main()

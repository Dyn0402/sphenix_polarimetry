#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 29 11:01 AM 2024
Created in PyCharm
Created as sphenix_polarimetry/plot_cad_measurements.py

@author: Dylan Neff, Dylan
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def main():
    file_path = 'C:/Users/Dylan/Desktop/vernier_scan/VernierScan_Aug12_crossing_angle.dat'
    data = read_crossing_angle(file_path)
    plot_crossing_angle(data)
    print('donzo')


def plot_crossing_angle(data):
    plt.figure(figsize=(12, 6))

    plt.plot(data['time'], data['bh8_crossing_angle'], label='Blue', color='blue')
    plt.plot(data['time'], data['yh8_crossing_angle'], label='Yellow', color='orange')
    plt.plot(data['time'], data['gh8_crossing_angle'], label='Blue - Yellow', color='green')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Angle (mrad)')
    plt.title('Crossing Angles vs Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def read_crossing_angle(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header lines
    data_lines = lines[3:]

    # Lists to store the parsed data
    time_data = []
    bh8_crossing_angle = []
    yh8_crossing_angle = []
    gh8_crossing_angle = []

    # Parse each line
    for line in data_lines:
        columns = line.strip().split('\t')
        if len(columns) != 4:
            continue

        time_data.append(datetime.strptime(columns[0].strip(), "%m/%d/%Y %H:%M:%S"))
        bh8_crossing_angle.append(float(columns[1]))
        yh8_crossing_angle.append(float(columns[2]))
        gh8_crossing_angle.append(float(columns[3]))

    # Return the data as a dictionary
    return {
        'time': time_data,
        'bh8_crossing_angle': bh8_crossing_angle,
        'yh8_crossing_angle': yh8_crossing_angle,
        'gh8_crossing_angle': gh8_crossing_angle,
    }


if __name__ == '__main__':
    main()

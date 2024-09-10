#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 29 11:01 AM 2024
Created in PyCharm
Created as sphenix_polarimetry/plot_cad_measurements.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pytz
from babel.dates import time_
from decorator import append
from sympy.polys.benchmarks.bench_groebnertools import time_vertex_color_12_vertices_23_edges


def main():
    # vernier_scan_date = 'Aug12'
    vernier_scan_date = 'Jul11'
    cad_measurements_path = 'C:/Users/Dylan/Desktop/vernier_scan/CAD_Measurements/'
    # cad_measurements_path = '/local/home/dn277127/Bureau/vernier_scan/CAD_Measurements/'
    # crossing_angle(cad_measurements_path, vernier_scan_date)
    # bunch_length(cad_measurements_path, vernier_scan_date)
    # beam_offset_and_intensity(cad_measurements_path, vernier_scan_date)
    combine_cad_measurements(cad_measurements_path, vernier_scan_date)
    plt.show()
    print('donzo')


def combine_cad_measurements(cad_measurements_path, vernier_scan_date):
    crossing_angle_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_crossing_angle.dat'
    bunch_length_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_bunch_length.dat'
    beam_offset_intensity_x_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_x.dat'
    beam_offset_intensity_y_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_y.dat'

    bunch_length_data = read_bunch_length(bunch_length_path)
    convert_bunch_length_to_distance(bunch_length_data)

    beam_offset_intensity_x_data = read_beam_offset_and_intensity(beam_offset_intensity_x_path)
    beam_offset_intensity_x_data['orientation'] = 'Horizontal'
    beam_offset_intensity_x_data = append_relative_and_set_offsets(beam_offset_intensity_x_data)
    beam_offset_intensity_y_data = read_beam_offset_and_intensity(beam_offset_intensity_y_path)
    beam_offset_intensity_y_data['orientation'] = 'Vertical'
    beam_offset_intensity_y_data = append_relative_and_set_offsets(beam_offset_intensity_y_data)
    boi_data = pd.concat([beam_offset_intensity_x_data, beam_offset_intensity_y_data], axis=0, ignore_index=True)

    crossing_angle_data = read_crossing_angle(crossing_angle_path)
    summarized_crossing_angle_data = average_crossing_angles(crossing_angle_data, bunch_length_data['time'])

    print(summarized_crossing_angle_data['time'])
    print(bunch_length_data['time'])

    summarized_crossing_angle_data = summarized_crossing_angle_data.set_index('time')
    bunch_length_data = bunch_length_data.set_index('time')
    boi_data = boi_data.set_index('time')

    combined_data = pd.concat([summarized_crossing_angle_data, bunch_length_data, boi_data], axis=1)
    combined_data = combined_data.reset_index()
    print(combined_data)
    print(combined_data.columns)
    print(combined_data.iloc[0])

    combined_data.to_csv(f'{cad_measurements_path}VernierScan_{vernier_scan_date}_combined.dat', sep='\t')


def crossing_angle(cad_measurements_path, vernier_scan_date):
    file_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_crossing_angle.dat'
    data = read_crossing_angle(file_path)
    plot_crossing_angle(data)


def bunch_length(cad_measurements_path, vernier_scan_date):
    file_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_bunch_length.dat'
    data = read_bunch_length(file_path)
    convert_bunch_length_to_distance(data)
    plot_bunch_length(data)


def beam_offset_and_intensity(cad_measurements_path, vernier_scan_date):
    file_path_x = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_x.dat'
    file_path_y = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_y.dat'
    data_x = read_beam_offset_and_intensity(file_path_x)
    plot_beam_offset_and_intensity(data_x, 'Horizontal')
    data_y = read_beam_offset_and_intensity(file_path_y)
    plot_beam_offset_and_intensity(data_y, 'Vertical')


def plot_crossing_angle(data):
    plt.figure(figsize=(12, 6))

    plt.plot(list(data['time']), list(data['bh8_crossing_angle']), label='Blue', color='blue')
    plt.plot(list(data['time']), list(data['yh8_crossing_angle']), label='Yellow', color='orange')
    plt.plot(list(data['time']), list(data['gh8_crossing_angle']), label='Blue - Yellow', color='green')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Angle (mrad)')
    plt.title('Crossing Angles vs Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_bunch_length(data):
    plt.figure(figsize=(12, 6))

    plt.plot(list(data['time']), list(data['blue_bunch_length']), label='Blue', color='blue', marker='o')
    plt.plot(list(data['time']), list(data['yellow_bunch_length']), label='Yellow', color='orange', marker='o')

    plt.xlabel('Time')
    plt.ylabel('Bunch Length (m)')
    plt.title('Bunch Length vs Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_beam_offset_and_intensity(data, orientation):
    fig_intensity, ax_intensity = plt.subplots(figsize=(12, 6))

    ax_intensity.plot(list(data['time']), list(data['wcm_blue']), label='WCM Blue', color='blue', marker='o')
    ax_intensity.plot(list(data['time']), list(data['wcm_yellow']), label='WCM Yellow', color='orange', marker='o')
    ax_intensity.plot(list(data['time']), list(data['dcct_blue']), label='DCCT Blue', color='blue', ls='--', marker='o')
    ax_intensity.plot(list(data['time']), list(data['dcct_yellow']), label='DCCT Yellow', color='orange', ls='--',
                      marker='o')

    ax_intensity.set_xlabel('Time')
    ax_intensity.set_ylabel('Intensity (10^9 protons)')
    ax_intensity.set_title(f'{orientation} Scan Beam Intensity vs Time')
    ax_intensity.legend()
    ax_intensity.grid(True)
    ax_intensity.tick_params(axis='x', rotation=45)
    fig_intensity.tight_layout()

    fig_offset, ax_offset = plt.subplots(figsize=(12, 6))

    ax_offset.plot(list(data['time']), list(data['bpm_south_ir']), label='BPM South IR', color='black', marker='o')
    ax_offset.plot(list(data['time']), list(data['bpm_north_ir']), label='BPM North IR', color='green', marker='o')
    ax_offset.axhline(data['bpm_south_ir'][0], color='black', linestyle='-', alpha=0.7)
    ax_offset.axhline(data['bpm_north_ir'][0], color='green', linestyle='-', alpha=0.7)

    ax_offset.set_xlabel('Time')
    ax_offset.set_ylabel('Offset (mm)')
    ax_offset.set_title(f'{orientation} Scan Beam Offset vs Time')
    ax_offset.legend()
    ax_offset.grid(True)
    ax_offset.tick_params(axis='x', rotation=45)
    fig_offset.tight_layout()

    fig_offset_from_baseline, ax_offset_from_baseline = plt.subplots(figsize=(12, 6))

    ax_offset_from_baseline.plot(list(data['time']), list(data['bpm_south_ir'] - data['bpm_south_ir'][0]),
                                 label='BPM South IR', color='black', marker='o')
    ax_offset_from_baseline.plot(list(data['time']), list(data['bpm_north_ir'] - data['bpm_north_ir'][0]),
                                 label='BPM North IR', color='green', marker='o')
    ax_offset_from_baseline.axhline(0, color='gray', linestyle='-', alpha=0.7)

    ax_offset_from_baseline.set_xlabel('Time')
    ax_offset_from_baseline.set_ylabel('Offset from Baseline (mm)')
    ax_offset_from_baseline.set_title(f'{orientation} Scan Beam Offset from Baseline vs Time')
    ax_offset_from_baseline.legend()
    ax_offset_from_baseline.grid(True)
    ax_offset_from_baseline.tick_params(axis='x', rotation=45)
    fig_offset_from_baseline.tight_layout()


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

    df = pd.DataFrame({
        'time': time_data,
        'bh8_crossing_angle': bh8_crossing_angle,
        'yh8_crossing_angle': yh8_crossing_angle,
        'gh8_crossing_angle': gh8_crossing_angle
    })

    # Set the time column to be in New York time
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_localize(bnl_tz)

    # Return the data as a pandas DataFrame
    return df

def read_bunch_length(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0)
    # Split the DataFrame into two parts: first 4 columns and last 4 columns
    n_actual_columns = df.shape[1] // 2
    first_part = df.iloc[:, :n_actual_columns]
    last_part = df.iloc[:, n_actual_columns:]

    # Rename the columns of the last part to match the first part
    last_part.columns = first_part.columns

    # Concatenate the two DataFrames vertically
    df = pd.concat([last_part, first_part], axis=0).reset_index(drop=True)

    df.columns = ['time', 'blue_bunch_length', 'yellow_time', 'yellow_bunch_length']
    df = df.drop(columns='yellow_time')
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_convert(bnl_tz)

    return df


def read_beam_offset_and_intensity(file_path):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=0)

    # Rename the columns
    cols = [
        'time',             # Time in Unix epoch seconds
        'wcm_blue',         # Wall Current Monitor for Blue beam [10^9 protons]
        'wcm_yellow',       # Wall Current Monitor for Yellow beam [10^9 protons]
        'dcct_blue',        # Direct Current-Current Transformer for Blue beam [10^9 protons]
        'dcct_yellow',      # Direct Current-Current Transformer for Yellow beam [10^9 protons]
        'bpm_south_ir',     # BPM Position in South side of IR [mm]
        'bpm_north_ir'      # BPM Position in North side of IR [mm]
    ]

    if len(df.columns) == 9:
        cols.insert(6, 'bpm_south_ir_sd')
        cols.insert(8, 'bpm_north_ir_sd')

    df.columns = cols

    # Convert the time from Unix epoch seconds to New York time
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_convert(bnl_tz)

    return df


def average_crossing_angles(crossing_angle_data, times):
    """
    Average crossing angle data over a given time range. Return averages along with standard deviations and mins/maxes.
    :param crossing_angle_data: DataFrame of crossing angle data.
    :param times: Pandas Series of time points to average around.
    :return: DataFrame of summarized crossing angle data.
    """
    # Get time start and end points from times Series using midpoints between points
    time_starts = times - (times - times.shift()) / 2
    time_ends = times + (times - times.shift()) / 2
    time_starts.iloc[0] = times.iloc[0] - (times.iloc[1] - times.iloc[0]) / 2
    time_ends.iloc[0] = times.iloc[0] + (times.iloc[1] - times.iloc[0]) / 2

    # Initialize lists to store averages, standard deviations, mins, and maxes
    avg_bh8, avg_yh8, avg_gh8, std_bh8, std_yh8, std_gh8 = [], [], [], [], [], []
    min_bh8, min_yh8, min_gh8, max_bh8, max_yh8, max_gh8 = [], [], [], [], [], []

    # Loop through time ranges and calculate averages, standard deviations, mins, and maxes
    for start, end in zip(time_starts, time_ends):
        data = crossing_angle_data[(crossing_angle_data['time'] >= start) & (crossing_angle_data['time'] <= end)]
        avg_bh8.append(data['bh8_crossing_angle'].mean())
        avg_yh8.append(data['yh8_crossing_angle'].mean())
        avg_gh8.append(data['gh8_crossing_angle'].mean())
        std_bh8.append(data['bh8_crossing_angle'].std())
        std_yh8.append(data['yh8_crossing_angle'].std())
        std_gh8.append(data['gh8_crossing_angle'].std())
        min_bh8.append(data['bh8_crossing_angle'].min())
        min_yh8.append(data['yh8_crossing_angle'].min())
        min_gh8.append(data['gh8_crossing_angle'].min())
        max_bh8.append(data['bh8_crossing_angle'].max())
        max_yh8.append(data['yh8_crossing_angle'].max())
        max_gh8.append(data['gh8_crossing_angle'].max())

    # Return the summarized data as a DataFrame
    return pd.DataFrame({
        'time': times,
        'time_start': time_starts,
        'time_end': time_ends,
        'bh8_avg': avg_bh8,
        'yh8_avg': avg_yh8,
        'gh8_avg': avg_gh8,
        'bh8_std': std_bh8,
        'yh8_std': std_yh8,
        'gh8_std': std_gh8,
        'bh8_min': min_bh8,
        'yh8_min': min_yh8,
        'gh8_min': min_gh8,
        'bh8_max': max_bh8,
        'yh8_max': max_yh8,
        'gh8_max': max_gh8
    })


def convert_bunch_length_to_distance(data):
    # Convert bunch length from fwhm in ns to 1 sigma in meters
    c = 299792458  # m/s
    data['blue_bunch_length'] = data['blue_bunch_length'] * c * 1e-9 / (2 * np.sqrt(2 * np.log(2)))
    data['yellow_bunch_length'] = data['yellow_bunch_length'] * c * 1e-9 / (2 * np.sqrt(2 * np.log(2)))


def append_relative_and_set_offsets(boi_data):
    """
    Append the relative and intended offsets to the beam offset and intensity data DataFrame.
    The relative offsets are the offsets relative to the first measured offset.
    The intended offsets are the offsets that are set in the control room. Use the relative offsets and match to the
    closest set_vals.
    """
    set_vals = [0.0, 0.1, 0.25, 0.4, 0.6, 0.9]

    # Create a copy of the DataFrame
    new_boi_data = boi_data.copy()

    # Calculate the relative offsets
    new_boi_data['bpm_south_ir_rel'] = new_boi_data['bpm_south_ir'] - new_boi_data['bpm_south_ir'].iloc[0]
    new_boi_data['bpm_north_ir_rel'] = new_boi_data['bpm_north_ir'] - new_boi_data['bpm_north_ir'].iloc[0]

    # Initialize new columns for the average and set values
    new_boi_data['offset_avg_val'] = new_boi_data['bpm_south_ir']
    new_boi_data['offset_set_val'] = new_boi_data['bpm_south_ir']

    # Calculate the average measured offsets and match to the closest set values
    for i in range(0, len(new_boi_data)):
        avg_meas_offset = (new_boi_data['bpm_south_ir_rel'].iloc[i] + new_boi_data['bpm_north_ir_rel'].iloc[i]) / 2
        closest_set_val = min(set_vals, key=lambda x: abs(x - abs(avg_meas_offset)))

        if closest_set_val != 0.0:
            closest_set_val *= np.sign(avg_meas_offset)

        new_boi_data.at[i, 'offset_avg_val'] = avg_meas_offset
        new_boi_data.at[i, 'offset_set_val'] = closest_set_val

    return new_boi_data


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 23 10:57 2024
Created in PyCharm
Created as sphenix_polarimetry/vernier_z_vertex_fitting

@author: Dylan Neff, dn277127
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
import pandas as pd

import uproot
import awkward as ak
import vector

from BunchCollider import BunchCollider
from Measure import Measure


def main():
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    # base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    dist_root_file_name = f'vernier_scan_{vernier_scan_date}_mbd_vertex_z_distributions.root'
    z_vertex_root_path = f'{base_path}vertex_data/{dist_root_file_name}'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    pdf_out_path = f'{base_path}/Analysis/sim_vs_mbd_cad_params_{vernier_scan_date}.pdf'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    # z_vertex_root_path = f'C:/Users/Dylan/Desktop/vernier_scan/vertex_data/{dist_root_file_name}'
    # fit_head_on(z_vertex_root_path)
    # fit_head_on_manual(z_vertex_root_path)
    # plot_head_on(z_vertex_root_path, longitudinal_fit_path)
    # head_on_metric_sensitivity(base_path, z_vertex_root_path, longitudinal_fit_path)
    plot_peripheral(z_vertex_root_path, longitudinal_fit_path)
    # fit_peripheral(z_vertex_root_path)
    # fit_peripheral_scipy(z_vertex_root_path, longitudinal_fit_path)
    # plot_head_on_and_peripheral(z_vertex_root_path, longitudinal_fit_path)
    # peripheral_metric_test(z_vertex_root_path)
    # peripheral_metric_sensitivity(base_path, z_vertex_root_path)
    # check_head_on_dependences(z_vertex_root_path)
    # plot_all_z_vertex_hists(z_vertex_root_path)
    # sim_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path)
    # sim_fit_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path)
    # perform_and_compare_vernier_scan(z_vertex_root_path, longitudinal_fit_path)

    print('donzo')


def perform_and_compare_vernier_scan(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path):
    """
    Perform a full vernier scan analysis and compare to MBD data.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    # Important parameters
    bw_nom = 170
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level

    # Will be overwritten by CAD values
    y_offset_nom = 0.
    angle_y_blue, angle_y_yellow = -0.05e-3, -0.18e-3

    collider_sim = BunchCollider()

    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)


def sim_fit_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path):
    """
    Run simulation with CAD measurements and fit to MBD distributions.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    print(cad_data)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    # Important parameters
    bw_nom = 135
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level

    # Will be overwritten by CAD values
    y_offset_nom = +750.
    angle_nom = +0.14e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    # Sort z_vertex_hists by scan axis and step, horizontal first
    z_vertex_hists = sorted(z_vertex_hists, key=lambda x: (x['scan_axis'], int(x['scan_step'])))

    for hist_data in z_vertex_hists:
        scan_orientation = hist_data['scan_axis']
        step_cad_data = cad_data[cad_data['orientation'] == scan_orientation].iloc[int(hist_data['scan_step'])]
        print(f'\nOrientation: {hist_data["scan_axis"]}, Step: {hist_data["scan_step"]}')
        print(step_cad_data)

        # Blue from left to right, yellow from right to left
        # Negative angle moves blue bunch from positive value to negative value, yellow from negative to positive
        # Offset blue bunch, fix yellow bunch at (0, 0)
        # Angle in x axis --> horizontal, negative angle from cad goes from negative to positive, flip of my convention
        # Horizontal scan in x, vertical scan in y
        xing_uncert = 50.  # microradians
        offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
        blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
        # blue_angle_min, blue_angle_max = -step_cad_data['bh8_min'] / 1e3, -step_cad_data['bh8_max'] / 1e3
        blue_angle_min, blue_angle_max = blue_angle - xing_uncert * 1e-6, blue_angle + xing_uncert * 1e-6
        # yellow_angle_min, yellow_angle_max = -step_cad_data['yh8_min'] / 1e3, -step_cad_data['yh8_max'] / 1e3
        yellow_angle_min, yellow_angle_max = yellow_angle - xing_uncert * 1e-6, yellow_angle + xing_uncert * 1e-6
        blue_bunch_len, yellow_bunch_len = step_cad_data['blue_bunch_length'], step_cad_data['yellow_bunch_length']
        blue_bunch_len, yellow_bunch_len = blue_bunch_len * 1e6, yellow_bunch_len * 1e6  # m to microns

        print(f'Offset: {offset}, Blue Angle: {blue_angle}, Yellow Angle: {yellow_angle}, Blue Bunch Length: {blue_bunch_len}, Yellow Bunch Length: {yellow_bunch_len}')

        if scan_orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif scan_orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

        collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
        blue_bunch_sigma = np.array([bw_nom, bw_nom, blue_bunch_len])
        yellow_bunch_sigma = np.array([bw_nom, bw_nom, yellow_bunch_len])
        collider_sim.set_bunch_sigmas(blue_bunch_sigma, yellow_bunch_sigma)

        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim_parallel()

        fit_amp_shift(collider_sim, hist_data['counts'], hist_data['centers'])
        zs, z_dist = collider_sim.get_z_density_dist()

        resid = np.sum((hist_data['counts'] - interp1d(zs, z_dist)(hist_data['centers'])) ** 2)
        print(f'Residual: {resid}')

        res = minimize(fit_beam_pars2, np.array([1.0, 1.0]),
                       args=(collider_sim, blue_angle, yellow_angle, hist_data['counts'], hist_data['centers']),
                       bounds=((0.0, 4.0), (0.0, 4.0)))
        print(res)
        angle1_y, angle2_y = res.x[0] * blue_angle, res.x[1] * yellow_angle
        # angle1_x, angle2_x = res.x[2] * blue_angle + 0., res.x[3] * yellow_angle + 0.

        # collider_sim.set_bunch_crossing(angle1_x, angle1_y, angle2_x, angle2_y)
        collider_sim.set_bunch_crossing(0., angle1_y, 0., angle2_y)
        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim_parallel()

        fit_amp_shift(collider_sim, hist_data['counts'], hist_data['centers'])
        zs_opt, z_dist_opt = collider_sim.get_z_density_dist()
        collider_param = collider_sim.get_param_string()

        fig, ax = plt.subplots(figsize=(8, 7))
        bin_width = hist_data['centers'][1] - hist_data['centers'][0]
        ax.bar(hist_data['centers'], hist_data['counts'], width=bin_width, label='MBD Vertex')
        ax.plot(zs, z_dist, color='g', alpha=0.5, ls='--', label='Simulation CAD Angles')
        ax.plot(zs_opt, z_dist_opt, color='r', label='Simulation Fit Angles')
        ax.set_xlim(-399, 399)
        ax.set_title(f'{hist_data["scan_axis"]} Scan Step {hist_data["scan_step"]} | {offset} um')
        ax.set_xlabel('z Vertex Position (cm)')
        ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(f"{pdf_out_path.replace('.pdf', f'_{scan_orientation}_{offset}.png')}", format='png')
        # plt.show()

    with PdfPages(pdf_out_path) as pdf:
        for fig_num in plt.get_fignums():
            # plt.savefig(plt.figure(fig_num), format='png')
            pdf.savefig(plt.figure(fig_num))
            plt.close(fig_num)


def sim_cad_params(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path):
    """
    Run simulation with CAD measurements and compare to MBD distributions.
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    print(cad_data)

    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    # Important parameters
    bw_nom = 135
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level

    # Will be overwritten by CAD values
    y_offset_nom = +750.
    bl1_nom = 130.e4
    bl2_nom = 117.e4
    angle_nom = +0.14e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    # Sort z_vertex_hists by scan axis and step, horizontal first
    z_vertex_hists = sorted(z_vertex_hists, key=lambda x: (x['scan_axis'], int(x['scan_step'])))

    for hist_data in z_vertex_hists:
        scan_orientation = hist_data['scan_axis']
        step_cad_data = cad_data[cad_data['orientation'] == scan_orientation].iloc[int(hist_data['scan_step'])]
        print(f'\nOrientation: {hist_data["scan_axis"]}, Step: {hist_data["scan_step"]}')
        print(step_cad_data)

        # Blue from left to right, yellow from right to left
        # Negative angle moves blue bunch from positive value to negative value, yellow from negative to positive
        # Offset blue bunch, fix yellow bunch at (0, 0)
        # Angle in x axis --> horizontal, negative angle from cad goes from negative to positive, flip of my convention
        # Horizontal scan in x, vertical scan in y
        xing_uncert = 50.  # microradians
        offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
        blue_angle, yellow_angle = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
        # blue_angle_min, blue_angle_max = -step_cad_data['bh8_min'] / 1e3, -step_cad_data['bh8_max'] / 1e3
        blue_angle_min, blue_angle_max = blue_angle - xing_uncert * 1e-6, blue_angle + xing_uncert * 1e-6
        # yellow_angle_min, yellow_angle_max = -step_cad_data['yh8_min'] / 1e3, -step_cad_data['yh8_max'] / 1e3
        yellow_angle_min, yellow_angle_max = yellow_angle - xing_uncert * 1e-6, yellow_angle + xing_uncert * 1e-6
        blue_bunch_len, yellow_bunch_len = step_cad_data['blue_bunch_length'], step_cad_data['yellow_bunch_length']
        blue_bunch_len, yellow_bunch_len = blue_bunch_len * 1e6, yellow_bunch_len * 1e6  # m to microns

        print(f'Offset: {offset}, Blue Angle: {blue_angle}, Yellow Angle: {yellow_angle}, Blue Bunch Length: {blue_bunch_len}, Yellow Bunch Length: {yellow_bunch_len}')

        if scan_orientation == 'Horizontal':
            collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
        elif scan_orientation == 'Vertical':
            collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

        collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
        blue_bunch_sigma = np.array([bw_nom, bw_nom, blue_bunch_len])
        yellow_bunch_sigma = np.array([bw_nom, bw_nom, yellow_bunch_len])
        collider_sim.set_bunch_sigmas(blue_bunch_sigma, yellow_bunch_sigma)

        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim_parallel()

        zs, z_dist = collider_sim.get_z_density_dist()
        scale = max(hist_data['counts']) / max(z_dist)
        z_max_sim = zs[np.argmax(z_dist)]
        z_max_hist = hist_data['centers'][np.argmax(hist_data['counts'])]
        shift = z_max_sim - z_max_hist  # microns

        res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                       args=(collider_sim, scale, shift, hist_data['counts'], hist_data['centers']),
                       bounds=((0.0, 2.0), (-10e4, 10e4)))
        scale = res.x[0] * scale
        shift = res.x[1] + shift

        collider_sim.set_amplitude(scale)
        collider_sim.set_z_shift(shift)
        zs, z_dist = collider_sim.get_z_density_dist()

        collider_param = collider_sim.get_param_string()

        resid = np.sum((hist_data['counts'] - interp1d(zs, z_dist)(hist_data['centers'])) ** 2)

        min_z_dist, max_z_dist = get_min_max_angles(collider_sim, hist_data, hist_data['centers'],
                                                    blue_angle_min, blue_angle_max, yellow_angle_min, yellow_angle_max)

        fig, ax = plt.subplots(figsize=(8, 7))
        bin_width = hist_data['centers'][1] - hist_data['centers'][0]
        ax.bar(hist_data['centers'], hist_data['counts'], width=bin_width, label='MBD Vertex')
        ax.plot(zs, z_dist, color='r', label='Simulation')
        ax.fill_between(hist_data['centers'], min_z_dist, max_z_dist, color='r', alpha=0.3)
        ax.set_xlim(-399, 399)
        ax.set_title(f'{hist_data["scan_axis"]} Scan Step {hist_data["scan_step"]} | {offset} um')
        ax.set_xlabel('z Vertex Position (cm)')
        ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top')
        ax.legend(loc='upper right')
        fig.tight_layout()
        # plt.show()

    with PdfPages(pdf_out_path) as pdf:
        for fig_num in plt.get_fignums():
            # plt.savefig(plt.figure(fig_num), format='png')
            pdf.savefig(plt.figure(fig_num))
            plt.close(fig_num)


def get_min_max_angles(collider_sim, hist_data, def_zs,
                       min_blue_x_angle, max_blue_x_angle, min_yellow_x_angle, max_yellow_x_angle):
    """
    Run combinations of blue and yellow crossing angles and get z_vertex distributions for each. From these create
    interp_1Ds. Finally, combine these to find the minimum and maximum z_vertex distributions.
    """
    z_dists = []
    for blue_angle in [min_blue_x_angle, max_blue_x_angle]:
        for yellow_angle in [min_yellow_x_angle, max_yellow_x_angle]:
            collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)
            collider_sim.set_amplitude(1.0)
            collider_sim.set_z_shift(0.0)
            collider_sim.run_sim_parallel()

            zs, z_dist = collider_sim.get_z_density_dist()
            scale = max(hist_data['counts']) / max(z_dist)
            z_max_sim = zs[np.argmax(z_dist)]
            z_max_hist = hist_data['centers'][np.argmax(hist_data['counts'])]
            shift = z_max_sim - z_max_hist  # microns

            res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                           args=(collider_sim, scale, shift, hist_data['counts'], hist_data['centers']),
                           bounds=((0.0, 2.0), (-10e4, 10e4)))
            scale = res.x[0] * scale
            shift = res.x[1] + shift

            collider_sim.set_amplitude(scale)
            collider_sim.set_z_shift(shift)
            zs, z_dist = collider_sim.get_z_density_dist()
            z_dists.append(interp1d(zs, z_dist)(def_zs))

    min_z_dist = np.min(z_dists, axis=0)
    max_z_dist = np.max(z_dists, axis=0)
    return min_z_dist, max_z_dist



def read_cad_measurement_file(cad_measurement_path):
    """
    Read CAD measurement file and return pandas data frame. First row is header, first column is index.
    """
    cad_data = pd.read_csv(cad_measurement_path, sep='\t', header=0, index_col=0)
    return cad_data


def check_head_on_dependences(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path)
    hist_0 = z_vertex_hists[0]

    # collider_sim = BunchCollider()
    # x_sim = np.array([150., 1.1e6, 85.])
    # bunch_widths = np.linspace(100, 200, 30)
    # resids = []
    # for bunch_width in bunch_widths:
    #     x_sim[0] = bunch_width
    #     res = fit_sim_pars_to_vertex(x_sim, hist_0['centers'], hist_0['counts'], collider_sim)
    #     resids.append(res)
    #
    # fig, ax = plt.subplots()
    # ax.plot(bunch_widths, resids)
    # ax.set_xlabel('Bunch Width (microns)')
    # ax.set_ylabel('Residual')

    collider_sim = BunchCollider()
    x_sim = np.array([150., 110, 85.])
    bunch_lengths = np.linspace(50, 200, 30)
    resids = []
    for bunch_length in bunch_lengths:
        x_sim[1] = bunch_length
        res = fit_sim_pars_to_vertex(x_sim, hist_0['centers'], hist_0['counts'], collider_sim, True)
        resids.append(res)

    fig, ax = plt.subplots()
    ax.plot(bunch_lengths / 1e6, resids)
    ax.set_xlabel('Bunch Length (m)')
    ax.set_ylabel('Residual')

    plt.show()


def fit_head_on(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path)
    hist_0 = z_vertex_hists[0]

    bin_width = hist_0['centers'][1] - hist_0['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(-0.13e-3, +0.0e-3)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    collider_param_str = collider_sim.get_param_string()

    # x0_sim = np.array([150., 110, 85.])
    # bounds = ((0, None), (0, None), (0, None))
    # x0_sim = np.array([117., 130., 85., 85.])
    # bounds = ((0, None), (0, None), (0, None), (0, None))
    x0_sim = np.array([117., 130.])
    bounds = ((1, None), (1, None))
    res = minimize(fit_sim_pars_to_vertex, x0_sim, args=(hist_0['centers'], hist_0['counts'], collider_sim, True),
                   bounds=bounds)
    print(res)

    x0_sim = np.array([*res.x, 85., 85.])
    bounds = ((0, None), (0, None), (0, None), (0, None))
    res = minimize(fit_sim_pars_to_vertex, x0_sim, args=(hist_0['centers'], hist_0['counts'], collider_sim, True),
                   bounds=bounds)
    print(res)

    collider_sim.run_sim()
    zs_opt, z_dist_opt = collider_sim.get_z_density_dist()
    collider_param_str_opt = collider_sim.get_param_string()

    scale = max(hist_0['counts']) / max(z_dist)
    scale_opt = max(hist_0['counts'] / max(z_dist_opt))

    fig, ax = plt.subplots()
    ax.bar(hist_0['centers'], hist_0['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs, z_dist * scale, color='r', label='Simulation')
    ax.plot(zs_opt, z_dist_opt * scale_opt, color='g', label='Simulation Optimized')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *x0), color='gray', label='Guess')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *res.x), color='green', label='Fit')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.bar(hist_0['centers'], hist_0['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_opt, z_dist_opt * scale_opt, color='r', label='Simulation Optimized')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *x0), color='gray', label='Guess')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *res.x), color='green', label='Fit')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str_opt}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def fit_head_on_manual(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path)
    hist_0 = z_vertex_hists[0]

    bin_width = hist_0['centers'][1] - hist_0['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(-0.13e-3, +0.0e-3)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    collider_param_str = collider_sim.get_param_string()

    scale = max(hist_0['counts']) / max(z_dist)

    fig, ax = plt.subplots()
    ax.bar(hist_0['centers'], hist_0['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs, z_dist * scale, color='r', label='Simulation')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def plot_head_on(z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)

    # step_to_use = 'vertical_0'
    step_to_use = 'vertical_6'

    all_params_dict = {
        'vertical_0': {
            'hist_num': 0,
            'angle_y_blue': -0.071e-3,
            'angle_y_yellow': -0.113e-3,
            'blue_len_scaling': 1.001144,
            'yellow_len_scaling': 1.00105,
            'rate': 845710.0067
        },
        'vertical_6': {
            'hist_num': 6,
            'angle_y_blue': -0.074e-3,
            'angle_y_yellow': -0.110e-3,
            'blue_len_scaling': 1.003789,
            'yellow_len_scaling': 1.004832,
            'rate': 843673.7041
        },
    }
    params = all_params_dict[step_to_use]

    hist = z_vertex_hists[params['hist_num']]
    # print(f'Time / scale_factor * correction factor: {np.sum(hist["counts"]) / params["rate"]}')
    time_per_step = 45  # seconds nominal time to spend per step
    # Adjust hist['counts'] to match rate for time_per_step
    hist['counts'] = hist['counts'] * params['rate'] / np.sum(hist['counts']) * time_per_step

    bin_width = hist['centers'][1] - hist['centers'][0]

    # Important parameters
    bw_nom = 160
    beta_star_nom = 85.
    mbd_online_resolution_nom = 5  # cm MBD resolution on trigger level
    # mbd_online_resolution_nom = 25.0  # cm MBD resolution on trigger level
    z_eff_width = 500.  # cm
    y_offset_nom = -0.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = params['yellow_len_scaling'], params['blue_len_scaling']
    angle_y_blue, angle_y_yellow = params['angle_y_blue'], params['angle_y_yellow']
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(z_eff_width)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    new_y_offset = 0.
    new_x_offset = 0.
    new_bw = 160
    beta_star = 110
    new_angle_y_blue, new_angle_y_yellow = params['angle_y_blue'], params['angle_y_yellow']
    # new_angle_y_blue, new_angle_y_yellow = -0.0e-3, -0.0e-3
    # new_angle_y_blue, new_angle_y_yellow = -0.07e-3, -0.114e-3
    # new_angle_x_blue, new_angle_x_yellow = 0.05e-3, -0.02e-3
    # new_angle_x_blue, new_angle_x_yellow = -0.05e-3, 0.05e-3
    new_angle_x_blue, new_angle_x_yellow = -0.0e-3, 0.0e-3
    # new_mbd_online_resolution_nom = None  # cm MBD resolution on trigger level
    new_mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    new_z_eff_width = 500.  # cm

    collider_sim.set_bunch_rs(np.array([new_x_offset, new_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(np.array([new_bw, new_bw]), np.array([new_bw, new_bw]))
    collider_sim.set_bunch_crossing(new_angle_x_blue, new_angle_y_blue, new_angle_x_yellow, new_angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(new_mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(new_z_eff_width)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift
    print(f'scale: {scale}, shift: {shift}')

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    residual = np.sum((hist['counts'] - interp1d(zs, z_dist)(hist['centers']))**2)
    print(f'Opt_shift_residual: {residual}')

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_xlim(-399, 399)
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def plot_peripheral(z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    rate = 8730.4249
    # time_correction_head_on = 0.23213476851091536
    # time_correction = np.sum(hist['counts']) / rate
    # print(f'Time / scale_factor * correction factor: {time_correction}')

    time_per_step = 45  # seconds nominal time to spend per step
    # Adjust hist['counts'] to match rate for time_per_step
    hist['counts'] = hist['counts'] * rate / np.sum(hist['counts']) * time_per_step

    bin_width = hist['centers'][1] - hist['centers'][0]

    # Important parameters
    bw_nom = 180
    beta_star_nom = 85.
    mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    z_eff_width = 500.  # cm
    y_offset_nom = -930.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = 1.00043315069711, 1.00030348127217
    angle_y_blue, angle_y_yellow = -0.07e-3, -0.114e-3
    angle_x_blue, angle_x_yellow = 0., 0.
    # fixed_scale = None
    # fixed_scale = 9.681147189701276e+28
    fixed_scale = 6.0464621817851715e+28

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(z_eff_width)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # microns
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    if fixed_scale is not None:
        scale = fixed_scale
    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    new_y_offset = -930
    new_x_offset = -100
    new_bw = 160.0
    new_beta_star = 85.0
    new_angle_y_blue, new_angle_y_yellow = -0.05e-3, -0.18e-3
    new_angle_x_blue, new_angle_x_yellow = 0.05e-3, -0.02e-3
    # new_angle_x_blue, new_angle_x_yellow = 0.0e-3, -0.0e-3
    new_mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    new_z_eff_width = 500.  # cm

    collider_sim.set_bunch_rs(np.array([new_x_offset, new_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(new_beta_star, new_beta_star)
    collider_sim.set_bunch_sigmas(np.array([new_bw, new_bw]), np.array([new_bw, new_bw]))
    collider_sim.set_bunch_crossing(new_angle_x_blue, new_angle_y_blue, new_angle_x_yellow, new_angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(new_mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(new_z_eff_width)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift
    print(f'scale: {scale}, shift: {shift}')

    if fixed_scale is not None:
        scale = fixed_scale
    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    residual = np.sum((hist['counts'] - interp1d(zs, z_dist)(hist['centers']))**2)
    print(f'Opt_shift_residual: {residual}')

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_xlim(-399, 399)
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def plot_head_on_and_peripheral(z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)

    # step_to_use = 'vertical_0'
    head_on_step_to_use = 'vertical_6'

    all_params_dict = {
        'vertical_0': {
            'hist_num': 0,
            'angle_y_blue': -0.071e-3,
            'angle_y_yellow': -0.113e-3,
            'blue_len_scaling': 1.001144,
            'yellow_len_scaling': 1.00105,
            'rate': 845710.0067
        },
        'vertical_6': {
            'hist_num': 6,
            'angle_y_blue': -0.074e-3,
            'angle_y_yellow': -0.110e-3,
            'blue_len_scaling': 1.003789,
            'yellow_len_scaling': 1.004832,
            'rate': 843673.7041
        },
        'horizontal_11': {
            'hist_num': -1,

        }
    }
    head_on_params = all_params_dict[head_on_step_to_use]

    head_on_hist = z_vertex_hists[head_on_params['hist_num']]
    # print(f'Time / scale_factor * correction factor: {np.sum(hist["counts"]) / params["rate"]}')
    time_per_step = 45  # seconds nominal time to spend per step
    # Adjust hist['counts'] to match rate for time_per_step
    head_on_hist['counts'] = head_on_hist['counts'] * head_on_params['rate'] / np.sum(head_on_hist['counts']) * time_per_step

    bin_width = head_on_hist['centers'][1] - head_on_hist['centers'][0]

    # Important parameters
    bw_nom = 160
    beta_star_nom = 85.
    mbd_online_resolution_nom = 25.0  # cm MBD resolution on trigger level
    y_offset_nom = -0.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = head_on_params['yellow_len_scaling'], head_on_params['blue_len_scaling']
    angle_y_blue, angle_y_yellow = head_on_params['angle_y_blue'], head_on_params['angle_y_yellow']
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(head_on_hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = head_on_hist['centers'][np.argmax(head_on_hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, head_on_hist['counts'], head_on_hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    new_y_offset = 0.
    new_x_offset = 0.
    new_bw = 160
    beta_star = 85
    new_angle_y_blue, new_angle_y_yellow = -0.07e-3, -0.114e-3
    # new_angle_x_blue, new_angle_x_yellow = 0.05e-3, -0.02e-3
    new_angle_x_blue, new_angle_x_yellow = -0.05e-3, 0.05e-3
    new_mbd_online_resolution_nom = 25.0  # cm MBD resolution on trigger level

    collider_sim.set_bunch_rs(np.array([new_x_offset, new_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(np.array([new_bw, new_bw]), np.array([new_bw, new_bw]))
    collider_sim.set_bunch_crossing(new_angle_x_blue, new_angle_y_blue, new_angle_x_yellow, new_angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(new_mbd_online_resolution_nom)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(head_on_hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = head_on_hist['centers'][np.argmax(head_on_hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, head_on_hist['counts'], head_on_hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift
    print(f'scale: {scale}, shift: {shift}')

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    residual = np.sum((head_on_hist['counts'] - interp1d(zs, z_dist)(head_on_hist['centers']))**2)
    print(f'Opt_shift_residual: {residual}')

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(head_on_hist['centers'], head_on_hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_xlim(-399, 399)
    ax.set_title(f'{head_on_hist["scan_axis"]} Scan Step {head_on_hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.01, 0.99), xycoords='axes fraction', verticalalignment='top',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def fit_peripheral(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]

    bin_width = hist['centers'][1] - hist['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., +750., -6.e6]), np.array([0., 0., +6.e6]))
    # collider_sim.set_bunch_rs(np.array([0., -1000., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    # collider_sim.set_bunch_sigmas(np.array([150., 150., 130.e4]), np.array([150., 150., 130.e4]))
    # collider_sim.set_bunch_crossing(-0.08e-3 / 2, +0.107e-3 / 2)
    # collider_sim.set_bunch_crossing(-0.07e-3 / 2, +0.114e-3 / 2)
    # collider_sim.set_bunch_crossing(-0.05e-3, 0)
    # collider_sim.set_bunch_crossing(-0.028e-3, 0)
    collider_sim.set_bunch_crossing(0, -0.2e-3, 0, +0.0e-3)
    # z_shift = -5  # cm Distance to shift center of collisions

    collider_sim.run_sim()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # microns
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate residual
    residual = np.sum((hist['counts'] - interp1d(zs_og, z_dist_og)(hist['centers']))**2)
    print(f'Guess Residual: {residual}')

    angles_y = np.linspace(-0.2e-3, -0.1e-3, 50)
    resids = []
    for angle_y in angles_y:
        collider_sim.set_bunch_crossing(0, angle_y, 0, +0.0e-3)
        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)
        collider_sim.run_sim()

        zs, z_dist = collider_sim.get_z_density_dist()
        scale = max(hist['counts']) / max(z_dist)
        z_max_sim = zs[np.argmax(z_dist)]
        z_max_hist = hist['centers'][np.argmax(hist['counts'])]
        shift = z_max_sim - z_max_hist  # microns

        res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                       args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                       bounds=((0.0, 2.0), (-10e4, 10e4)))
        scale = res.x[0] * scale
        shift = res.x[1] + shift

        collider_sim.set_amplitude(scale)
        collider_sim.set_z_shift(shift)
        zs, z_dist = collider_sim.get_z_density_dist()

        resid = np.sum((hist['counts'] - interp1d(zs, z_dist)(hist['centers']))**2)
        resids.append(resid)
        print(f'angle_y: {angle_y}, residual: {resid}')

        # fig, ax = plt.subplots()
        # ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
        # ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
        # ax.plot(zs, z_dist, color='r', label='Simulation Fit')
        # ax.set_title(f'Angle Y: {angle_y * 1e3:.4f} mrad')
        # ax.set_xlabel('z Vertex Position (cm)')
        # # ax.annotate(f'{collider_param}', (0.02, 0.75), xycoords='axes fraction',
        # #             bbox=dict(facecolor='wheat', alpha=0.5))
        # ax.annotate(f'Residual: {resid:.2e}', (0.02, 0.75), xycoords='axes fraction',
        #             bbox=dict(facecolor='wheat', alpha=0.5))
        # ax.legend()
        # fig.tight_layout()

    fig_resids, ax_resids = plt.subplots()
    ax_resids.plot(angles_y, resids)
    ax_resids.set_xlabel('Crossing Angle (rad)')
    ax_resids.set_ylabel('Residual')

    # Get best angle
    best_angle = angles_y[np.argmin(resids)]

    collider_sim.set_bunch_crossing(0, best_angle, 0, +0.0e-3)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist  # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def fit_peripheral_scipy(z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]

    bin_width = hist['centers'][1] - hist['centers'][0]

    # Important parameters
    bw_nom = 160
    beta_star_nom = 85.
    mbd_online_resolution = 5.0  # cm MBD resolution on trigger level
    y_offset_nom = -930.
    angle_y_blue, angle_y_yellow = -0.05e-3, -0.18e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    collider_sim.set_bkg(0.2e-17)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    shift *= 1e4  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate residual
    residual = np.sum((hist['counts'] - interp1d(zs_og, z_dist_og)(hist['centers']))**2)
    print(f'Guess Residual: {residual}')

    res = minimize(fit_beam_pars3, np.array([1.0, 1.0, 1.0]),
                   args=(collider_sim, angle_y_blue, angle_y_yellow, bw_nom,
                         hist['counts'], hist['centers']),
                   bounds=((0.1, 2.0), (0.1, 2.0), (0.1, 2.0)))
    print(res)
    angle_y_blue, angle_y_yellow = res.x[0] * angle_y_blue, res.x[1] * angle_y_yellow
    bw = res.x[2] * bw_nom

    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, angle_y_yellow)
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(hist['counts']) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim}, z_max_hist: {z_max_hist}')
    shift = z_max_sim - z_max_hist # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs, z_dist = collider_sim.get_z_density_dist()

    collider_param = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs_og, z_dist_og, color='gray', ls='--', alpha=0.6, label='Simulation Guess')
    ax.plot(zs, z_dist, color='r', label='Simulation Fit')
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.set_xlim(-349, 349)
    ax.annotate(f'{collider_param}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


def peripheral_metric_test(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    # hist = z_vertex_hists[0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., +750., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(0, -0.14e-3, 0, +0.0e-3)

    collider_sim.run_sim()
    zs_sim, z_dist_sim = collider_sim.get_z_density_dist()
    # fit_amp_shift(collider_sim, hist['counts'], hist['centers'])

    print('\nData Metrics:')
    metrics_data = get_dist_metrics(hist['centers'], hist['counts'], True)
    print('\nSimulation Metrics:')
    metrics_sim = get_dist_metrics(zs_sim, z_dist_sim, True)

    plt.show()


def peripheral_metric_sensitivity(base_path, z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    metrics_data = get_dist_metrics(hist['centers'], hist['counts'])

    out_dir = f'{base_path}Analysis/metric_sensitivities/'

    y_offset_nom = +750.
    bw_nom = 135
    bl1_nom = 130.e4
    bl2_nom = 117.e4
    angle_nom = -0.14e-3
    beta_star_nom = 85.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)

    points_per_var = 30

    angles = np.linspace(-0.2e-3, -0.1e-3, points_per_var)
    angle_metrics = []
    for angle in angles:
        collider_sim.set_bunch_crossing(0, angle, 0, 0)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        angle_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Angle: {angle}, Metrics: {angle_metrics_i}')
        angle_metrics.append(angle_metrics_i)
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)

    plot_metric_sensitivities(angles * 1e3, np.array(angle_metrics), metrics_data, angle_nom * 1e3, 'Crossing Angle (mrad)')

    beam_widths = np.linspace(100, 200, points_per_var)
    beam_width_metrics = []
    for beam_width in beam_widths:
        bunch1_sigma, bunch2_sigma = collider_sim.get_beam_sigmas()
        bunch1_sigma = np.array([beam_width, beam_width, bunch1_sigma[2]])
        bunch2_sigma = np.array([beam_width, beam_width, bunch2_sigma[2]])
        collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        beam_width_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Beam width: {beam_width}, Metrics: {beam_width_metrics_i}')
        beam_width_metrics.append(beam_width_metrics_i)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))

    plot_metric_sensitivities(beam_widths, np.array(beam_width_metrics), metrics_data, bw_nom, 'Beam Width (microns)')

    beta_stars = np.linspace(80, 90, points_per_var)
    beta_star_metrics = []
    for beta_star in beta_stars:
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        beta_star_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Beta Star: {beta_star}, Metrics: {beta_star_metrics_i}')
        beta_star_metrics.append(beta_star_metrics_i)
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)

    plot_metric_sensitivities(beta_stars, np.array(beta_star_metrics), metrics_data, beta_star_nom, 'Beta Star (cm)')

    y_offsets = np.linspace(700, 800, points_per_var)
    y_offset_metrics = []
    for y_offset in y_offsets:
        collider_sim.set_bunch_offsets(np.array([0., y_offset]), np.array([0., 0.]))
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        y_offset_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'Y Offset: {y_offset}, Metrics: {y_offset_metrics_i}')
        y_offset_metrics.append(y_offset_metrics_i)
    collider_sim.set_bunch_offsets(np.array([0., y_offset_nom]), np.array([0., 0.]))

    plot_metric_sensitivities(y_offsets, np.array(y_offset_metrics), metrics_data, y_offset_nom, 'Y Offset (cm)')

    bl1s = np.linspace(120.e4, 140.e4, points_per_var)
    bl1_metrics = []
    for bl1 in bl1s:
        bunch1_sigma, bunch2_sigma = collider_sim.get_beam_sigmas()
        bunch1_sigma = np.array([bw_nom, bw_nom, bl1])
        collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
        collider_sim.run_sim()
        zs, z_dist = collider_sim.get_z_density_dist()
        bl1_metrics_i = get_dist_metrics(zs, z_dist)
        print(f'BL1: {bl1}, Metrics: {bl1_metrics_i}')
        bl1_metrics.append(bl1_metrics_i)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))

    plot_metric_sensitivities(bl1s / 1e4, np.array(bl1_metrics), metrics_data, bl1_nom / 1e4, 'Bunch1 Length (cm)')

    for fig in plt.get_fignums():
        fig_title = plt.figure(fig).canvas.manager.get_window_title()
        plt.figure(fig).savefig(f'{out_dir}{fig_title}.png')

    plt.show()


def head_on_metric_sensitivity(base_path, z_vertex_root_path, longitudinal_fit_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[0]
    metrics_data = get_head_on_dist_width(hist['centers'], hist['counts'])

    out_dir = f'{base_path}Analysis/metric_sensitivities_head_on/'
    if not os.path.exists(out_dir):  # Make out_dir if it doesn't exist
        os.makedirs(out_dir)

    # Important parameters
    bw_nom = 180
    beta_star_nom = 85.
    mbd_online_resolution_nom = 5.0  # cm MBD resolution on trigger level
    y_offset_nom = -0.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = 0.991593955543314, 0.993863022403956
    angle_y_blue, angle_y_yellow = -0.07e-3, -0.114e-3
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    points_per_var = 30

    angles = np.linspace(-0.2e-3, -0.0e-3, points_per_var)
    angle_metrics = []
    for angle in angles:
        collider_sim.set_bunch_crossing(0, angle, 0, 0)
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        angle_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Angle: {angle}, Metrics: {angle_metrics_i}')
        angle_metrics.append(angle_metrics_i)
    collider_sim.set_bunch_crossing(0, angle_y_blue, 0, 0)

    plot_metric_sensitivities(angles * 1e3, np.array(angle_metrics), metrics_data, angle_y_blue * 1e3, 'Crossing Angle (mrad)')

    beam_widths = np.linspace(100, 200, points_per_var)
    beam_width_metrics = []
    for beam_width in beam_widths:
        bunch1_sigma = np.array([beam_width, beam_width])
        bunch2_sigma = np.array([beam_width, beam_width])
        collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        beam_width_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Beam width: {beam_width}, Metrics: {beam_width_metrics_i}')
        beam_width_metrics.append(beam_width_metrics_i)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))

    plot_metric_sensitivities(beam_widths, np.array(beam_width_metrics), metrics_data, bw_nom, 'Beam Width (microns)')

    beta_stars = np.linspace(80, 90, points_per_var)
    beta_star_metrics = []
    for beta_star in beta_stars:
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        beta_star_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Beta Star: {beta_star}, Metrics: {beta_star_metrics_i}')
        beta_star_metrics.append(beta_star_metrics_i)
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)

    plot_metric_sensitivities(beta_stars, np.array(beta_star_metrics), metrics_data, beta_star_nom, 'Beta Star (cm)')

    y_offsets = np.linspace(-100, 100, points_per_var)
    y_offset_metrics = []
    for y_offset in y_offsets:
        collider_sim.set_bunch_offsets(np.array([0., y_offset]), np.array([0., 0.]))
        collider_sim.run_sim_parallel()
        zs, z_dist = collider_sim.get_z_density_dist()
        y_offset_metrics_i = get_head_on_dist_width(zs, z_dist)
        print(f'Y Offset: {y_offset}, Metrics: {y_offset_metrics_i}')
        y_offset_metrics.append(y_offset_metrics_i)
    collider_sim.set_bunch_offsets(np.array([0., y_offset_nom]), np.array([0., 0.]))

    plot_metric_sensitivities(y_offsets, np.array(y_offset_metrics), metrics_data, y_offset_nom, 'Y Offset (cm)')

    # bl1s = np.linspace(120.e4, 140.e4, points_per_var)
    # bl1_metrics = []
    # for bl1 in bl1s:
    #     bunch1_sigma, bunch2_sigma = collider_sim.get_beam_sigmas()
    #     bunch1_sigma = np.array([bw_nom, bw_nom, bl1])
    #     collider_sim.set_bunch_sigmas(bunch1_sigma, bunch2_sigma)
    #     collider_sim.run_sim()
    #     zs, z_dist = collider_sim.get_z_density_dist()
    #     bl1_metrics_i = get_dist_metrics(zs, z_dist)
    #     print(f'BL1: {bl1}, Metrics: {bl1_metrics_i}')
    #     bl1_metrics.append(bl1_metrics_i)
    # collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bl1_nom]), np.array([bw_nom, bw_nom, bl2_nom]))
    #
    # plot_metric_sensitivities(bl1s / 1e4, np.array(bl1_metrics), metrics_data, bl1_nom / 1e4, 'Bunch1 Length (cm)')

    for fig in plt.get_fignums():
        fig_title = plt.figure(fig).canvas.manager.get_window_title()
        plt.figure(fig).savefig(f'{out_dir}{fig_title}.png')

    plt.show()


def plot_metric_sensitivities(variable_vals, metrics, metrics_data, var_nom, variable_xlabel=None):
    var_name = variable_xlabel.split('(')[0].strip()
    fig_height_ratio, ax_hr = plt.subplots()
    ax_hr.errorbar(variable_vals, [m.val for m in metrics[:, 0]], yerr=[m.err for m in metrics[:, 0]],
                   label='Height Ratio', ls='none', marker='o')
    ax_hr.set_xlabel(variable_xlabel)
    ax_hr.set_ylabel('Height Ratio')
    ax_hr.axhline(metrics_data[0].val, color='green', ls='--', label='Data')
    ax_hr.axhspan(metrics_data[0].val - metrics_data[0].err, metrics_data[0].val + metrics_data[0].err,
                    color='green', alpha=0.3)
    ax_hr.axvline(var_nom, color='blue', ls='--', label='Nominal')
    ax_hr2 = ax_hr.twinx()
    ax_hr2.plot([], [])
    ax_hr2.set_ylabel('Height Ratio % Diff')
    ax_hr2.set_ylim((np.array(ax_hr.get_ylim()) - metrics_data[0].val) / metrics_data[0].val * 100)
    ax_hr2.yaxis.set_major_formatter(PercentFormatter())
    ax_hr.legend()
    ax_hr.set_title(f'Height Ratio Sensitivity to {var_name}')
    fig_height_ratio.canvas.manager.set_window_title(f'Height Ratio Sensitivity to {var_name}')
    fig_height_ratio.tight_layout()

    if metrics.shape[1] < 3:
        return

    fig_peak_sep, ax_ps = plt.subplots()
    ax_ps.errorbar(variable_vals, [m.val for m in metrics[:, 1]], yerr=[m.err for m in metrics[:, 1]],
                   label='Peak Separation', ls='none', marker='o')
    ax_ps.set_xlabel(variable_xlabel)
    ax_ps.set_ylabel('Peak Separation (cm)')
    ax_ps.axhline(metrics_data[1].val, color='green', ls='--', label='Data')
    ax_ps.axhspan(metrics_data[1].val - metrics_data[1].err, metrics_data[1].val + metrics_data[1].err,
                    color='green', alpha=0.3)
    ax_ps.axvline(var_nom, color='blue', ls='--', label='Nominal')
    ax_ps2 = ax_ps.twinx()
    ax_ps2.plot([], [])
    ax_ps2.set_ylabel('Peak Separation % Diff')
    ax_ps2.set_ylim((np.array(ax_ps.get_ylim()) - metrics_data[1].val) / metrics_data[1].val * 100)
    ax_ps2.yaxis.set_major_formatter(PercentFormatter())
    ax_ps.legend()
    ax_ps.set_title(f'Peak Separation Sensitivity to {var_name}')
    fig_peak_sep.canvas.manager.set_window_title(f'Peak Separation Sensitivity to {var_name}')
    fig_peak_sep.tight_layout()

    fig_main_peak_width, ax_mpw = plt.subplots()
    ax_mpw.errorbar(variable_vals, [m.val for m in metrics[:, 2]], yerr=[m.err for m in metrics[:, 2]],
                    label='Main Peak Width', ls='none', marker='o')
    ax_mpw.set_xlabel(variable_xlabel)
    ax_mpw.set_ylabel('Main Peak Width (cm)')
    ax_mpw.axhline(metrics_data[2].val, color='green', ls='--', label='Data')
    ax_mpw.axhspan(metrics_data[2].val - metrics_data[2].err, metrics_data[2].val + metrics_data[2].err,
                    color='green', alpha=0.3)
    ax_mpw.axvline(var_nom, color='blue', ls='--', label='Nominal')
    ax_mpw2 = ax_mpw.twinx()
    ax_mpw2.plot([], [])
    ax_mpw2.set_ylabel('Main Peak Width % Diff')
    ax_mpw2.set_ylim((np.array(ax_mpw.get_ylim()) - metrics_data[2].val) / metrics_data[2].val * 100)
    ax_mpw2.yaxis.set_major_formatter(PercentFormatter())
    ax_mpw.legend()
    ax_mpw.set_title(f'Main Peak Width Sensitivity to {var_name}')
    fig_main_peak_width.canvas.manager.set_window_title(f'Main Peak Width Sensitivity to {var_name}')
    fig_main_peak_width.tight_layout()


def get_dist_metrics(zs, z_dist, plot=False):
    """
    Get metrics characterizing the given z-vertex distribution.
    Height ratio: Ratio of the heights of the two peaks.
    Peak separation: Distance between the peaks.
    Main peak width: Width of the main peak.
    :param zs: cm z-vertex positions
    :param z_dist: z-vertex distribution
    :param plot: True to plot the distribution and fit
    :return: height_ratio, peak_separation, main_peak_width
    """
    bin_width = zs[1] - zs[0]

    max_hist = np.max(z_dist)
    z_max_hist = zs[np.argmax(z_dist)]

    # If z_max_hist > 0 find max on negative side, else find max on positive side
    if z_max_hist > 0:
        second_counts = z_dist[zs < 0]
        second_zs = zs[zs < 0]
    else:
        second_counts = z_dist[zs > 0]
        second_zs = zs[zs > 0]

    max_second = np.max(second_counts)
    z_max_second = second_zs[np.argmax(second_counts)]

    sigma_est = 50

    p0 = [max_hist, z_max_hist, sigma_est, max_second, z_max_second, sigma_est]

    popt, pcov = cf(double_gaus_bkg, zs, z_dist, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    opt_measures = [Measure(p, e) for p, e in zip(popt, perr)]

    height_ratio = opt_measures[0] / opt_measures[3]
    peak_separation = abs(opt_measures[1] - opt_measures[4])
    main_peak_width = opt_measures[2]

    if plot:
        fig, ax = plt.subplots()
        ax.bar(zs, z_dist, width=bin_width, label='MBD Vertex')
        ax.plot(zs, double_gaus_bkg(zs, *p0), color='gray', ls='--', alpha=0.6, label='Guess')
        ax.plot(zs, double_gaus_bkg(zs, *popt), color='r', label='Fit')
        ax.set_xlabel('z Vertex Position (cm)')
        text_str = (f'Height Ratio: {height_ratio}\nPeak Separation: {peak_separation} cm\n'
                    f'Main Peak Width: {main_peak_width} cm')
        print(f'p0: {p0}')
        opt_str = ', '.join([str(measure) for measure in opt_measures])
        print(f'Optimized: {opt_str}')
        print(text_str)
        ax.annotate(text_str, (0.02, 0.75), xycoords='axes fraction', bbox=dict(facecolor='wheat', alpha=0.5))
        fig.tight_layout()

    return height_ratio, peak_separation, main_peak_width


def get_head_on_dist_width(zs, z_dist, plot=False):
    """
    Get width of the main peak of the given z-vertex distribution.
    :param zs: cm z-vertex positions
    :param z_dist: z-vertex distribution
    :param plot: True to plot the distribution and fit
    :return: main_peak_width
    """
    bin_width = zs[1] - zs[0]

    max_hist = np.max(z_dist)
    z_max_hist = zs[np.argmax(z_dist)]

    sigma_est = 50

    p0 = [max_hist, z_max_hist, sigma_est]

    popt, pcov = cf(gaus_bkg, zs, z_dist, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    opt_measures = [Measure(p, e) for p, e in zip(popt, perr)]

    main_peak_width = opt_measures[2]

    if plot:
        fig, ax = plt.subplots()
        ax.bar(zs, z_dist, width=bin_width, label='MBD Vertex')
        ax.plot(zs, gaus_bkg(zs, *p0), color='gray', ls='--', alpha=0.6, label='Guess')
        ax.plot(zs, gaus_bkg(zs, *popt), color='r', label='Fit')
        ax.set_xlabel('z Vertex Position (cm)')
        text_str = f'Main Peak Width: {main_peak_width} cm'
        print(f'p0: {p0}')
        opt_str = ', '.join([str(measure) for measure in opt_measures])
        print(f'Optimized: {opt_str}')
        print(text_str)
        ax.annotate(text_str, (0.02, 0.75), xycoords='axes fraction', bbox=dict(facecolor='wheat', alpha=0.5))
        fig.tight_layout()

    return (main_peak_width,)


def get_mbd_z_dists(z_vertex_dist_root_path, first_dist=True):
    vector.register_awkward()

    z_vertex_hists = []
    with uproot.open(z_vertex_dist_root_path) as file:
        print(file.keys())
        for key in file.keys():
            hist = file[key]
            z_vertex_hists.append({
                'scan_axis': key.split('_')[1],
                'scan_step': key.split('_')[-1].split(';')[0],
                'centers': hist.axis().centers(),
                'counts': hist.counts(),
                'count_errs': hist.errors()
            })
            if first_dist:
                break
    return z_vertex_hists


def plot_all_z_vertex_hists(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    for hist in z_vertex_hists:
        fig, ax = plt.subplots()
        bin_width = hist['centers'][1] - hist['centers'][0]
        ax.bar(hist['centers'], hist['counts'], width=bin_width)
        ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
        fig.tight_layout()
    plt.show()


def fit_sim_pars_to_vertex(x, zs, z_dist, collider_sim, scale_fit=False):
    if len(x) == 2:
        length_1, length_2 = x
        length_1 *= 1e4
        length_2 *= 1e4
        bunch_sigmas_1, bunch_sigmas_2 = collider_sim.get_beam_sigmas()
        bunch_sigmas_1[2] = length_1
        bunch_sigmas_2[2] = length_2
        collider_sim.set_bunch_sigmas(bunch_sigmas_1, bunch_sigmas_2)
    elif len(x) == 3:
        width, length, beta_star = x
        length *= 1e4
        bunch_sigmas = np.array([width, width, length])
        collider_sim.set_bunch_sigmas(bunch_sigmas, bunch_sigmas)
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    elif len(x) == 4:
        length_1, length_2, beta_star_1, beta_star_2 = x
        length_1 *= 1e4
        length_2 *= 1e4
        bunch_sigmas_1 = np.array([150, 150, length_1])
        bunch_sigmas_2 = np.array([150, 150, length_2])
        collider_sim.set_bunch_sigmas(bunch_sigmas_1, bunch_sigmas_2)
        collider_sim.set_bunch_beta_stars(beta_star_1, beta_star_2)
    elif len(x) == 6:
        width_1, width_2, length_1, length_2, beta_star_1, beta_star_2 = x
        length_1 *= 1e4
        length_2 *= 1e4
        bunch_sigmas_1 = np.array([width_1, width_1, length_1])
        bunch_sigmas_2 = np.array([width_2, width_2, length_2])
        collider_sim.set_bunch_sigmas(bunch_sigmas_1, bunch_sigmas_2)
        collider_sim.set_bunch_beta_stars(beta_star_1, beta_star_2)

    collider_sim.run_sim()
    # collider_sim.run_sim_parallel()
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    if scale_fit:
        scale_0 = max(z_dist) / max(sim_z_dist)
        res = minimize(fit_amp, np.array([scale_0]), args=(sim_zs, sim_z_dist, z_dist, zs))
        scale = res.x
    else:
        scale = max(z_dist) / max(sim_z_dist)
    sim_interp = interp1d(sim_zs / 1e4, sim_z_dist * scale)
    residual = np.sum((z_dist - sim_interp(zs))**2)
    print(f'{x}: {residual}')
    return residual


def fit_beam_pars(x, collider_sim, angle1_y_0, offset1_y_0, beam_width_0, beta_star_0, beam1_length_0, beam2_length_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param offset1_y_0:
    # :param offset1_x0:
    :param beam_width_0:
    :param beta_star_0:
    :param beam1_length_0:
    :param beam2_length_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle_y = x[0] * angle1_y_0
    offset1_y = x[1] * offset1_y_0
    # offset1_x = x[2] * offset1_x0
    beam_width = x[2] * beam_width_0
    beta_star = x[3] * beta_star_0
    beam1_length = x[4] * beam1_length_0
    beam2_length = x[5] * beam2_length_0
    collider_sim.set_bunch_crossing(0, angle_y, 0, +0.0)
    bunch1_r = collider_sim.bunch1_r_original
    bunch2_r = collider_sim.bunch2_r_original
    # bunch1_r[1], bunch1_r[0] = offset1_y, offset1_x
    bunch1_r[1] = offset1_y
    collider_sim.set_bunch_rs(bunch1_r, bunch2_r)
    # bunch1_sigmas, bunch2_sigmas = collider_sim.get_beam_sigmas()
    bunch1_sigmas = np.array([beam_width, beam_width, beam1_length])
    bunch2_sigmas = np.array([beam_width, beam_width, beam2_length])
    collider_sim.set_bunch_sigmas(bunch1_sigmas, bunch2_sigmas)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim()

    fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = np.sum((z_dist_data - interp1d(zs, z_dist)(zs_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        print(f'angle_y_0: {angle1_y_0}, offset1_y_0: {offset1_y_0}, beam_width_0: {beam_width_0}, beta_star_0: {beta_star_0}, beam1_length_0: {beam1_length_0}, beam2_length_0: {beam2_length_0}')
        print(f'angle_y: {angle_y}, offset1_y: {offset1_y}, beam_width: {beam_width}, beta_star: {beta_star}, beam1_length: {beam1_length}, beam2_length: {beam2_length}')

    return residual


def fit_beam_pars2(x, collider_sim, angle1_y_0, angle2_y_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param angle2_y_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle1_y = x[0] * angle1_y_0
    # angle1_x = x[1] * angle1_y_0 + angle1_x_0
    angle2_y = x[1] * angle2_y_0
    # angle2_x = x[3] * angle2_y_0 + angle2_x_0
    collider_sim.set_bunch_crossing(0.0, angle1_y, 0.0, angle2_y)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim_parallel()

    fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = np.sum((z_dist_data - interp1d(zs, z_dist)(zs_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual


def fit_beam_pars3(x, collider_sim, angle1_y_0, angle2_y_0, bw_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param angle2_y_0:
    :param bw_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle1_y = x[0] * angle1_y_0
    # angle1_x = x[1] * angle1_y_0 + angle1_x_0
    angle2_y = x[1] * angle2_y_0
    # angle2_x = x[3] * angle2_y_0 + angle2_x_0
    bw = x[2] * bw_0
    collider_sim.set_bunch_crossing(0.0, angle1_y, 0.0, angle2_y)
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim_parallel()

    fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = np.sum((z_dist_data - interp1d(zs, z_dist)(zs_data)) ** 2)
    print(f'{x}: {residual:.2e}')
    if np.isnan(residual):
        print(zs)
        print(z_dist)
        # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual


def fit_beam_pars4(x, collider_sim, angle1_y_0, angle2_y_0, bw_0,
                  z_dist_data, zs_data):
    """
    Fit beam parameters
    :param x:
    :param collider_sim:
    :param angle1_y_0:
    :param angle2_y_0:
    :param bw_0:
    :param z_dist_data:
    :param zs_data:
    :return:
    """
    angle1_y = x[0] * angle1_y_0
    # angle1_x = x[1] * angle1_y_0 + angle1_x_0
    angle2_y = x[1] * angle2_y_0
    # angle2_x = x[3] * angle2_y_0 + angle2_x_0
    bw = x[2] * bw_0
    collider_sim.set_bunch_crossing(0.0, angle1_y, 0.0, angle2_y)
    collider_sim.set_bunch_sigmas(np.array([bw, bw]), np.array([bw, bw]))
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)
    collider_sim.run_sim_parallel()

    # fit_amp_shift(collider_sim, z_dist_data, zs_data)
    zs, z_dist = collider_sim.get_z_density_dist()
    residual = get_dist_metric_residuals(zs, z_dist, z_dist_data, zs_data)
    print(f'{x}: {residual}')
    # if np.isnan(residual):
    #     print(zs)
    #     print(z_dist)
    #     # print(f'angle1_y_0: {angle1_y_0}, angle1_x_0: {angle1_x_0}, angle2_y_0: {angle2_y_0}, angle2_x_0: {angle2_x_0}')

    return residual.val


def get_dist_metric_residuals(zs, z_dist, z_dist_data, zs_data):
    """
    Get residual between simulated and measured distribution metrics
    """
    metrics_data = get_dist_metrics(zs_data, z_dist_data)
    metrics_sim = get_dist_metrics(zs, z_dist)
    residuals = [((m_val - s_val) / m_val)**2 for m_val, s_val in zip(metrics_data, metrics_sim)]
    print(f'Residuals: {residuals}')
    residual = np.mean(residuals)
    return residual


def fit_amp_shift(collider_sim, z_dist_data, zs_data):
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(z_dist_data) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = zs_data[np.argmax(z_dist_data)]
    shift = z_max_sim - z_max_hist  # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, z_dist_data, zs_data),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)


def amp_shift_residual(x, collider_sim, scale_0, shift_0, z_dist_data, zs_data):
    collider_sim.set_amplitude(x[0] * scale_0)
    collider_sim.set_z_shift(x[1] + shift_0)
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    return np.sum((z_dist_data - sim_interp(zs_data)) ** 2)


def gaus(x, a, x0, sigma):
    return a * np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)


def double_gaus_bkg(x, a1, x01, sigma1, a2, x02, sigma2):
    return gaus(x, a1, x01, sigma1) + gaus(x, a2, x02, sigma2)


def gaus_bkg(x, a, x0, sigma):
    return gaus(x, a, x0, sigma)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 23 10:57 2024
Created in PyCharm
Created as sphenix_polarimetry/vernier_z_vertex_fitting

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d

import uproot
import awkward as ak
import vector

from BunchCollider import BunchCollider
from Measure import Measure


def main():
    # vernier_scan_date = 'Aug12'
    vernier_scan_date = 'Jul11'
    dist_root_file_name = f'vernier_scan_{vernier_scan_date}_mbd_vertex_z_distributions.root'
    z_vertex_root_path = f'/local/home/dn277127/Bureau/vernier_scan/vertex_data/{dist_root_file_name}'
    # z_vertex_root_path = f'C:/Users/Dylan/Desktop/vernier_scan/vertex_data/{dist_root_file_name}'
    # fit_head_on(z_vertex_root_path)
    # fit_head_on_manual(z_vertex_root_path)
    # plot_peripheral(z_vertex_root_path)
    # fit_peripheral(z_vertex_root_path)
    # fit_peripheral_scipy(z_vertex_root_path)
    # peripheral_metric_test(z_vertex_root_path)
    # peripheral_metric_sensitivity(z_vertex_root_path)
    # check_head_on_dependences(z_vertex_root_path)
    plot_all_z_vertex_hists(z_vertex_root_path)

    print('donzo')


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


def plot_peripheral(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]

    bin_width = hist['centers'][1] - hist['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., +750., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(0, -0.115e-3, 0, -0.075e-3)

    collider_sim.run_sim()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()
    # collider_param_str_og = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # microns
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10, 10)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    collider_sim.set_bunch_rs(np.array([+50., +750., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    beam_width = 135.
    collider_sim.set_bunch_sigmas(np.array([beam_width, beam_width, 130.e4]), np.array([beam_width, beam_width, 117.e4]))
    collider_sim.set_bunch_crossing(-0.01e-3, -0.125e-3, +0.01e-3, -0.07e-3)
    collider_sim.set_amplitude(1.0)
    collider_sim.set_z_shift(0.0)

    collider_sim.run_sim()
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
                   bounds=((0.0, 2.0), (-10, 10)))
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
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10, 10)))
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
                       bounds=((0.0, 2.0), (-10, 10)))
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
                   bounds=((0.0, 2.0), (-10, 10)))
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


def fit_peripheral_scipy(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]

    bin_width = hist['centers'][1] - hist['centers'][0]

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., +750., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(85., 85.)
    collider_sim.set_bunch_sigmas(np.array([135., 135., 130.e4]), np.array([135., 135., 117.e4]))
    collider_sim.set_bunch_crossing(0, -0.14e-3, 0, +0.0e-3)

    collider_sim.run_sim()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    scale = max(hist['counts']) / max(z_dist_og)
    z_max_sim_og = zs_og[np.argmax(z_dist_og)]
    z_max_hist_og = hist['centers'][np.argmax(hist['counts'])]
    print(f'z_max_sim: {z_max_sim_og}, z_max_hist: {z_max_hist_og}')
    shift = z_max_sim_og - z_max_hist_og  # cm
    print(f'scale: {scale}, shift: {shift}')

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]), args=(collider_sim, scale, shift, hist['counts'], hist['centers']),
                   bounds=((0.0, 2.0), (-10, 10)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate residual
    residual = np.sum((hist['counts'] - interp1d(zs_og, z_dist_og)(hist['centers']))**2)
    print(f'Guess Residual: {residual}')

    angle1_y_0 = -0.14e-3
    offset_y0 = +750.
    # offset_x0 = 0
    beam_width_0 = 135.
    beta_star_0 = 85.
    bl1_0 = 130.e4
    bl2_0 = 117.e4
    res = minimize(fit_beam_pars, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                   args=(collider_sim, angle1_y_0, offset_y0, beam_width_0, beta_star_0, bl1_0, bl2_0,
                         hist['counts'], hist['centers']),
                   bounds=((0.1, 2.0), (0.1, 2.0), (0.1, 2.0), (0.1, 2.0), (0.1, 2.0), (0.1, 2.0)))
    print(res)
    angle1_y, offset_y0 = res.x[0] * angle1_y_0, res.x[1] * offset_y0

    collider_sim.set_bunch_crossing(0, angle1_y, 0, 0.0)
    bunch1_r = collider_sim.bunch1_r_original
    bunch2_r = collider_sim.bunch2_r_original
    bunch1_r[1] = offset_y0
    collider_sim.set_bunch_rs(bunch1_r, bunch2_r)
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
                   bounds=((0.0, 2.0), (-10, 10)))
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


def peripheral_metric_sensitivity(z_vertex_root_path):
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, False)
    hist = z_vertex_hists[-1]
    metrics_data = get_dist_metrics(hist['centers'], hist['counts'])

    out_dir = '/local/home/dn277127/Bureau/vernier_scan/Analysis/metric_sensitivities/'

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


def fit_amp_shift(collider_sim, z_dist_data, zs_data):
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(z_dist_data) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = zs_data[np.argmax(z_dist_data)]
    shift = z_max_sim - z_max_hist  # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, z_dist_data, zs_data),
                   bounds=((0.0, 2.0), (-10, 10)))
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


if __name__ == '__main__':
    main()

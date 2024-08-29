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
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import uproot
import awkward as ak
import vector

from beam_beam_sim import BunchCollider


def main():
    # z_verted_root_path = '/local/home/dn277127/Bureau/vernier_scan/vertex_data/hist_out.root'
    z_verted_root_path = 'C:/Users/Dylan/Desktop/vernier_scan/vertex_data/hist_out.root'
    # fit_head_on(z_verted_root_path)
    # fit_head_on_manual(z_verted_root_path)
    fit_peripheral(z_verted_root_path)
    # check_head_on_dependences(z_verted_root_path)
    # plot_all_z_vertex_hists(z_verted_root_path)

    print('donzo')


def check_head_on_dependences(z_verted_root_path):
    z_vertex_hists = get_mbd_z_dists(z_verted_root_path)
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


def fit_head_on(z_verted_root_path):
    z_vertex_hists = get_mbd_z_dists(z_verted_root_path)
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
    ax.plot(zs / 1e4, z_dist * scale, color='r', label='Simulation')
    ax.plot(zs_opt / 1e4, z_dist_opt * scale_opt, color='g', label='Simulation Optimized')
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
    ax.plot(zs_opt / 1e4, z_dist_opt * scale_opt, color='r', label='Simulation Optimized')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *x0), color='gray', label='Guess')
    # ax.plot(hist_0['centers'], gaus(hist_0['centers'], *res.x), color='green', label='Fit')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str_opt}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def fit_head_on_manual(z_verted_root_path):
    z_vertex_hists = get_mbd_z_dists(z_verted_root_path)
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
    ax.plot(zs / 1e4, z_dist * scale, color='r', label='Simulation')
    ax.set_title(f'{hist_0["scan_axis"]} Scan Step {hist_0["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def fit_peripheral(z_verted_root_path):
    z_vertex_hists = get_mbd_z_dists(z_verted_root_path, False)
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
    collider_sim.set_bunch_crossing(-0.13e-3, +0.0e-3)

    collider_sim.run_sim()
    zs, z_dist = collider_sim.get_z_density_dist()
    collider_param_str = collider_sim.get_param_string()

    scale = max(hist['counts']) / max(z_dist)

    fig, ax = plt.subplots()
    ax.bar(hist['centers'], hist['counts'], width=bin_width, label='MBD Vertex')
    ax.plot(zs / 1e4, z_dist * scale, color='r', label='Simulation')
    ax.set_title(f'{hist["scan_axis"]} Scan Step {hist["scan_step"]}')
    ax.set_xlabel('z Vertex Position (cm)')
    ax.annotate(f'{collider_param_str}', (0.02, 0.75), xycoords='axes fraction',
                bbox=dict(facecolor='wheat', alpha=0.5))
    ax.legend()
    fig.tight_layout()

    plt.show()


def get_mbd_z_dists(z_vertex_dist_root_path, first_dist=True):
    vector.register_awkward()
    # z_vertex_dist_root_path = '/local/home/dn277127/Bureau/vernier_scan/vertex_data/hist_out.root'

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


def plot_all_z_vertex_hists(z_verted_root_path):
    z_vertex_hists = get_mbd_z_dists(z_verted_root_path, False)
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


def fit_amp(x, sim_zs, sim_z_dist, z_dist, zs):
    sim_interp = interp1d(sim_zs / 1e4, sim_z_dist * x[0])
    return np.sum((z_dist - sim_interp(zs))**2)


def gaus(x, a, x0, sigma):
    return a * np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)


if __name__ == '__main__':
    main()

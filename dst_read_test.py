#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 01 03:36 2024
Created in PyCharm
Created as sphenix_polarimetry/dst_read_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import vector


def main():
    # dst_path = '/local/home/dn277127/Bureau/test_prdf/DST_TRIGGERED_EVENT_run2pp_new_2024p003-00049764-00001.root'
    # dst_path = '/local/home/dn277127/Bureau/test_prdf/DST_TRIGGERED_EVENT_run2pp_new_2024p003-00049763-00080.root'
    dst_path = ('/local/home/dn277127/Bureau/vernier_scan/dst_data/'
                'DST_TRIGGERED_EVENT_run2pp_new_2024p003-00048029-0002.root')
    with uproot.open(dst_path) as file:
        print(file.keys())
        tree = file['T']
        print(tree.keys())
        scaler16 = tree['DST#GL1#GL1Packet/gl1pscaler[16][3]']
        scaler64 = tree['DST#GL1#GL1Packet/scaler[64][3]']
        bunch_num = tree['DST#GL1#GL1Packet/BunchNumber']
        bco = tree['DST#GL1#GL1Packet/Gl1Packet/OfflinePacketv1/bco']
        mbd = tree['DST#MBD#MBDPackets']
        print(mbd.keys())
        print(scaler16)
        # uproot scaler is a TBranchElement. It has a 'array' method that returns a jagged array.
        # The jagged array is an array of arrays, where the outer array is the number of events and the inner arrays
        # are the scalers for each event.
        # The jagged array is awkward array, so it can be accessed like a numpy array.
        print(scaler16.array())
        print(scaler64.array())
        print(np.array(scaler64.array()).shape)
        scaler64_array = np.array(scaler64.array())
        scaler16_array = np.array(scaler16.array())
        fig, ax = plt.subplots()
        for i in range(8):
            ax.plot(scaler16_array[:, i, 0], label=f'scaler {i}')
        ax.legend()
        fig.tight_layout()
        print(bunch_num.array())
        print(bco.array())
        print(len(scaler16.array()))
        print(len(scaler64.array()))
        print(len(bunch_num.array()))
        print(len(bco.array()))
        plt.show()
    print('donzo')


if __name__ == '__main__':
    main()

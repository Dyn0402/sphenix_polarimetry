#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 31 06:31 2024
Created in PyCharm
Created as sphenix_polarimetry/vernier_scan

@author: Dylan Neff, dn277127
"""


def main():
    scan_path = '/local/home/dn277127/Bureau/vernier_scan/'
    horizontal_file_name = 'sPhenix.WcmDcctBpm.x.34785.dat'
    vertical_file_name = 'sPhenix.WcmDcctBpm.y.34785.dat'
    with open(scan_path + horizontal_file_name, 'r') as file:
        horizontal_data = file.readlines()
    for line in horizontal_data:
        print(line)
    print('donzo')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Compute SNR for different freqs

Created on Sat Mar 16 11:02:14 2024

@author: mings
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import argparse

import tensorflow.keras.backend as K
    
from mmwchanmod.datasets.download import load_model 
from mmwchanmod.sim.antenna import Elem3GPP, ElemDipole
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import dir_path_loss, dir_path_loss_multi_sect
from mmwchanmod.common.constants import LinkState    

"""
Parse arguments from command line
"""

model_dir = './models/0225-6GHz-reduce70%'
model_name = model_dir.split('/')[-1]
"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)
    
"""
Antenna parameters
"""
# https://arxiv.org/abs/2309.03038 Fig.2(b) and Table.II
bw_ls = [100e6, 200e6, 300e6, 400e6] # bandwidth in Hz
fc_ls = [6e9, 12e9, 18e9, 24e9] # freq in Hz
nant_gnb_ls = [np.array([2, 2]), np.array([4, 4]), np.array([5, 5]), 
               np.array([7, 7])]
nant_ue_ls = [np.array([1, 2]), np.array([1, 2]), np.array([1, 3]), 
               np.array([1, 3])]
nsect = 3  # number of sectors for terrestrial gNBs 
nf = 7  # Noise figure in dB
kT = -174   # Thermal noise in dBm/Hz
tx_pow = 33  # gNB power in dBm
# downtilt = -12  # downtilt in degrees
downtilt = 0

"""
Create the arrays
"""
elem_3gpp = Elem3GPP(thetabw=65, phibw=65)
arr_gnb_ls = [URA(elem = elem_3gpp, nant = nant_gnb_ls[idx],
                  fc = fc_ls[idx]) for idx in range(len(fc_ls))]
arr_gnb_sects_ls = [multi_sect_array(arr_gnb_ls[idx], 
                                     sect_type = 'azimuth', 
                                     theta0 = downtilt, 
                                     nsect = nsect) 
                    for idx in range(len(fc_ls))]
arr_ue_ls = []
for ifc in range(len(fc_ls)):
    temp = URA(elem = elem_3gpp, nant = nant_ue_ls[ifc], fc = fc_ls[ifc])
    arr_ue_ls.append(RotatedArray(temp, theta0=90)) # point up


"""
Load the pre-trained model
"""
# Construct and load the channel model object
mod_name = 'Beijing'
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(model_name, cfg, 'local')

"""
Codebook design
"""
# TX codebook with az 15 degree  
phi_ls = [np.linspace(-45, 60, 8), 
          np.linspace(75, 180, 8), 
          np.linspace(-165, -60, 8)]
# theta = np.linspace(downtilt, downtilt, 8)
theta = np.linspace(0, 0, 8)
tx_codebook_ls = [] # len = nfreq
for ifreq in range(len(fc_ls)):
# for ifreq in range(1):
    tx_codebook_sub_arr = []
    for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
        phi_angles = phi_ls[itx]
        tx_codebook= []
        tx_sv_ls, tx_elem_gain = tx_arr.sv(phi_angles, theta,
                                         return_elem_gain=True)
        for ibf in range(len(tx_sv_ls)):
            tx_sv = tx_sv_ls[ibf]
            wtx = np.conj(tx_sv)
            wtx = wtx / np.sqrt(np.sum(np.abs(wtx)**2))
            tx_codebook.append(wtx)
            # tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
        tx_codebook_sub_arr.append(tx_codebook)
    tx_codebook_ls.append(tx_codebook_sub_arr)

"""
Generate chan_list for test data
"""
if_real, if_generated, if_plot = False, False, True
if if_real:
    # Using the ray tracing data (real path data)
    chan_list, ls = data_to_mpchan(test_data, cfg) 
    dvec = test_data['dvec']
    d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
    # Only considering links distance in (10, 500) meters
    dmin, dmax = 10, 150 # m
    I = np.where((d3d >= dmin) & (d3d <= dmax))[0].astype(int) 
    print('Total link', len(I))
    """
    Plot a random link
    """
    aoa_phi_ind = 0
    aoa_theta_ind = 1
    aod_phi_ind = 2
    aod_theta_ind = 3
    large_pl = 261
    
    
    best_snr_lower_freq_beam, best_snr_high_freq = [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            continue
        chan = chan_list[link_idx]
        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        for ifreq in range(3,4):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            best_snr_high_freq.append(best_snr)
            
        for ifreq in range(1):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            
        ifreq = 3
        rx_arr = arr_ue_ls[ifreq]
        tx_arr = arr_gnb_sects_ls[ifreq][best_sect]
        wtx = tx_codebook_ls[ifreq][best_sect][best_bf_direct]
        tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        tx_bf_ls, rx_bf_ls = [], []
        for ipath in range(len(aod_phi)):
            rx_sv = rx_svi[ipath]
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            rx_bf_ls.append(rx_bf)
            tx_sv = tx_svi[ipath] 
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            tx_bf_ls.append(tx_bf)
        pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
        temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
        best_snr_lower_freq_beam.append(temp_snr)
        # print(ifreq, best_snr, best_sect, best_bf_direct)
    
    with open('snr_data_24GHz_real.pkl', 'wb') as file:
        pickle.dump([best_snr_lower_freq_beam, best_snr_high_freq], file)
    
    best_snr_lower_freq_beam, best_snr_high_freq = [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            continue
        chan = chan_list[link_idx]
        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        for ifreq in range(2,3):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            best_snr_high_freq.append(best_snr)
            
        for ifreq in range(1):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            
        ifreq = 2
        rx_arr = arr_ue_ls[ifreq]
        tx_arr = arr_gnb_sects_ls[ifreq][best_sect]
        wtx = tx_codebook_ls[ifreq][best_sect][best_bf_direct]
        tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        tx_bf_ls, rx_bf_ls = [], []
        for ipath in range(len(aod_phi)):
            rx_sv = rx_svi[ipath]
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            rx_bf_ls.append(rx_bf)
            tx_sv = tx_svi[ipath] 
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            tx_bf_ls.append(tx_bf)
        pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
        temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
        best_snr_lower_freq_beam.append(temp_snr)
        # print(ifreq, best_snr, best_sect, best_bf_direct)
    
    with open('snr_data_18GHz_real.pkl', 'wb') as file:
        pickle.dump([best_snr_lower_freq_beam, best_snr_high_freq], file)
        
        
    best_snr_lower_freq_beam, best_snr_high_freq = [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            continue
        chan = chan_list[link_idx]
        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        for ifreq in range(1,2):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            best_snr_high_freq.append(best_snr)
            
        for ifreq in range(1):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            
        ifreq = 1
        rx_arr = arr_ue_ls[ifreq]
        tx_arr = arr_gnb_sects_ls[ifreq][best_sect]
        wtx = tx_codebook_ls[ifreq][best_sect][best_bf_direct]
        tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        tx_bf_ls, rx_bf_ls = [], []
        for ipath in range(len(aod_phi)):
            rx_sv = rx_svi[ipath]
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            rx_bf_ls.append(rx_bf)
            tx_sv = tx_svi[ipath] 
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            tx_bf_ls.append(tx_bf)
        pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
        temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
        best_snr_lower_freq_beam.append(temp_snr)
        # print(ifreq, best_snr, best_sect, best_bf_direct)
    
    with open('snr_data_12GHz_real.pkl', 'wb') as file:
        pickle.dump([best_snr_lower_freq_beam, best_snr_high_freq], file)



if if_generated:
    # Using the generated data (real path data)
    ls = test_data['link_state']
    fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
    fspl_ls = test_data['fspl_ls'].transpose(1,0)
    chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)
    dvec = test_data['dvec']
    d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)

    # Only considering links distance in (10, 500) meters
    dmin, dmax = 10, 150 # m
    I = np.where((d3d >= dmin) & (d3d <= dmax))[0].astype(int) 
    print('Total link', len(I))
    """
    Plot a random link
    """
    aoa_phi_ind = 0
    aoa_theta_ind = 1
    aod_phi_ind = 2
    aod_theta_ind = 3
    large_pl = 261
    
    
    best_snr_lower_freq_beam, best_snr_high_freq = [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            continue
        chan = chan_list[link_idx]
        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        for ifreq in range(3,4):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            best_snr_high_freq.append(best_snr)
            
        for ifreq in range(1):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            
        ifreq = 3
        rx_arr = arr_ue_ls[ifreq]
        tx_arr = arr_gnb_sects_ls[ifreq][best_sect]
        wtx = tx_codebook_ls[ifreq][best_sect][best_bf_direct]
        tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        tx_bf_ls, rx_bf_ls = [], []
        for ipath in range(len(aod_phi)):
            rx_sv = rx_svi[ipath]
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            rx_bf_ls.append(rx_bf)
            tx_sv = tx_svi[ipath] 
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            tx_bf_ls.append(tx_bf)
        pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
        temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
        best_snr_lower_freq_beam.append(temp_snr)
        # print(ifreq, best_snr, best_sect, best_bf_direct)
    
    with open('snr_data_24GHz_train.pkl', 'wb') as file:
        pickle.dump([best_snr_lower_freq_beam, best_snr_high_freq], file)
    
    best_snr_lower_freq_beam, best_snr_high_freq = [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            continue
        chan = chan_list[link_idx]
        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        for ifreq in range(2,3):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            best_snr_high_freq.append(best_snr)
            
        for ifreq in range(1):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            
        ifreq = 2
        rx_arr = arr_ue_ls[ifreq]
        tx_arr = arr_gnb_sects_ls[ifreq][best_sect]
        wtx = tx_codebook_ls[ifreq][best_sect][best_bf_direct]
        tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        tx_bf_ls, rx_bf_ls = [], []
        for ipath in range(len(aod_phi)):
            rx_sv = rx_svi[ipath]
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            rx_bf_ls.append(rx_bf)
            tx_sv = tx_svi[ipath] 
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            tx_bf_ls.append(tx_bf)
        pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
        temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
        best_snr_lower_freq_beam.append(temp_snr)
        # print(ifreq, best_snr, best_sect, best_bf_direct)
    
    with open('snr_data_18GHz_train.pkl', 'wb') as file:
        pickle.dump([best_snr_lower_freq_beam, best_snr_high_freq], file)
        
        
    best_snr_lower_freq_beam, best_snr_high_freq = [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            continue
        chan = chan_list[link_idx]
        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        for ifreq in range(1,2):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            best_snr_high_freq.append(best_snr)
            
        for ifreq in range(1):
            rx_arr = arr_ue_ls[ifreq]
            best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
            for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
                tx_codebook = tx_codebook_ls[ifreq][itx]
                tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                for iwtx, wtx in enumerate(tx_codebook):
                    tx_bf_ls, rx_bf_ls = [], []
                    for ipath in range(len(aod_phi)):
                        rx_sv = rx_svi[ipath]
                        wrx = np.conj(rx_sv)
                        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
                        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
                        rx_bf_ls.append(rx_bf)
                        tx_sv = tx_svi[ipath] 
                        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
                        tx_bf_ls.append(tx_bf)
                    pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
                    pl_min = np.min(pl_bf)
                    pl_lin = 10**(-0.1*(pl_bf-pl_min))
                    pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
                    temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
                    if temp_snr > best_snr:
                        best_snr = temp_snr
                        best_sect = itx
                        best_bf_direct = iwtx
            
        ifreq = 1
        rx_arr = arr_ue_ls[ifreq]
        tx_arr = arr_gnb_sects_ls[ifreq][best_sect]
        wtx = tx_codebook_ls[ifreq][best_sect][best_bf_direct]
        tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        tx_bf_ls, rx_bf_ls = [], []
        for ipath in range(len(aod_phi)):
            rx_sv = rx_svi[ipath]
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            rx_bf_ls.append(rx_bf)
            tx_sv = tx_svi[ipath] 
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            tx_bf_ls.append(tx_bf)
        pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
        temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
        best_snr_lower_freq_beam.append(temp_snr)
        # print(ifreq, best_snr, best_sect, best_bf_direct)
    
    with open('snr_data_12GHz_train.pkl', 'wb') as file:
        pickle.dump([best_snr_lower_freq_beam, best_snr_high_freq], file)


if if_plot:
    best_snr_lower_freq_beam_train, best_snr_high_freq_train = [], []
    best_snr_lower_freq_beam_real, best_snr_high_freq_real = [], []
    snr_lb, snr_ub = 0,26
    with open('snr_data_12GHz_train.pkl', 'rb') as file:
        temp = pickle.load(file)
        temp_lower, temp_higher = [], []
        for i in range(len(temp[0])):
            if snr_lb <= temp[0][i] <= snr_ub and snr_lb <= temp[1][i] <= snr_ub:
                temp_lower.append(temp[0][i])
                temp_higher.append(temp[1][i])
        best_snr_lower_freq_beam_train.append(temp_lower)
        best_snr_high_freq_train.append(temp_higher)
    with open('snr_data_18GHz_train.pkl', 'rb') as file:
        temp = pickle.load(file)
        temp_lower, temp_higher = [], []
        for i in range(len(temp[0])):
            if snr_lb <= temp[0][i] <= snr_ub and snr_lb <= temp[1][i] <= snr_ub:
                temp_lower.append(temp[0][i])
                temp_higher.append(temp[1][i])
        best_snr_lower_freq_beam_train.append(temp_lower)
        best_snr_high_freq_train.append(temp_higher)
    with open('snr_data_24GHz_train.pkl', 'rb') as file:
        temp = pickle.load(file)
        temp_lower, temp_higher = [], []
        for i in range(len(temp[0])):
            if snr_lb <= temp[0][i] <= snr_ub and snr_lb <= temp[1][i] <= snr_ub:
                temp_lower.append(temp[0][i])
                temp_higher.append(temp[1][i])
        best_snr_lower_freq_beam_train.append(temp_lower)
        best_snr_high_freq_train.append(temp_higher)
    with open('snr_data_12GHz_real.pkl', 'rb') as file:
        temp = pickle.load(file)
        temp_lower, temp_higher = [], []
        for i in range(len(temp[0])):
            if snr_lb <= temp[0][i] <= snr_ub :
                temp_lower.append(temp[0][i])
                temp_higher.append(temp[1][i])
        best_snr_lower_freq_beam_real.append(temp_lower)
        best_snr_high_freq_real.append(temp_higher)
    with open('snr_data_18GHz_real.pkl', 'rb') as file:
        temp = pickle.load(file)
        temp_lower, temp_higher = [], []
        for i in range(len(temp[0])):
            if snr_lb <= temp[0][i] <= snr_ub :
                temp_lower.append(temp[0][i])
                temp_higher.append(temp[1][i])
        best_snr_lower_freq_beam_real.append(temp_lower)
        best_snr_high_freq_real.append(temp_higher)
    with open('snr_data_24GHz_real.pkl', 'rb') as file:
        temp = pickle.load(file)
        temp_lower, temp_higher = [], []
        for i in range(len(temp[0])):
            if snr_lb <= temp[0][i] <= snr_ub :
                temp_lower.append(temp[0][i])
                temp_higher.append(temp[1][i])
        best_snr_lower_freq_beam_real.append(temp_lower)
        best_snr_high_freq_real.append(temp_higher)
    
    label_ls = ['12GHz', '18GHz', '24GHz'] 
    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1,3, figsize=(13,4))
    for i in range(3):
        error = np.array(best_snr_high_freq_real[i]) - np.array(best_snr_lower_freq_beam_real[i])
        ni = len(error)
        p = np.arange(ni)/ni
        ax[i].plot(np.sort(error), p, '-', color='blue')
        error = np.array(best_snr_high_freq_train[i]) - np.array(best_snr_lower_freq_beam_train[i])
        ni = len(error)
        p = np.arange(ni)/ni
        ax[i].plot(np.sort(error), p, '--', color='red')
        ax[i].set_ylim([.75, 1.0])
        ax[i].set_xlim([0, 20])
        ax[i].grid()
        ax[i].set_title(label_ls[i])
        if i == 0:
            ax[i].set_ylabel('CDF')
        else:
            ax[i].set_yticklabels([])  # 删除其他子图的y轴标签
        if i == 2:
            ax[i].legend(['Ray Tracing Samples', 'Generated Links'], loc='lower right')

            
    fig.text(0.5, 0.00, 'SNR Difference (dB) Between Selections at Higher Frequencies and 6GHz', ha='center')
    fig.tight_layout()
    plt.savefig('0225-6GHz-reduce70%-New/snr-beam-selected-cdf.png', dpi = 400)
    plt.show()

# -*- coding: utf-8 -*-
"""
Compute SNR for different freqs

Created on Sat Mar 16 11:02:14 2024

@author: mings
"""

import numpy as np
import matplotlib.pyplot as plt
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
# parser = argparse.ArgumentParser(description='Plots the SNR for diff freqs')  
# parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
#     help='directory to store models')
# parser.add_argument(\
#     '--plot_dir',action='store',\
#     default='plots', help='directory for the output plots')    
# parser.add_argument(\
#     '--plot_fn',action='store',\
#     default='snr_two_freq.png', help='plot file name')    
# args = parser.parse_args()
# model_dir = args.model_dir
# model_name = model_dir.split('/')[-1]
# plot_dir = args.plot_dir
# plot_fn = args.plot_fn

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

# Using the ray tracing data (real path data)
# chan_list, ls = data_to_mpchan(test_data, cfg) 
dvec = test_data['dvec']

d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)

ls = test_data['link_state']
fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
fspl_ls = test_data['fspl_ls'].transpose(1,0)
chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)


# Only considering links distance in (10, 500) meters
dmin, dmax = 300, 500 # m
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

# best_snr_freqs_ls = [[] for i in range(len(fc_ls))]
# for link_idx in I:
#     if chan_list[link_idx].link_state == LinkState.no_link:
#         continue
#     chan = chan_list[link_idx]
#     aod_theta = chan.ang[:,aod_theta_ind]
#     aod_phi = chan.ang[:,aod_phi_ind]
#     aoa_theta = chan.ang[:,aoa_theta_ind]
#     aoa_phi = chan.ang[:,aoa_phi_ind]
#     pl_min = large_pl
#     for ifreq in range(len(fc_ls)):
#         rx_arr = arr_ue_ls[ifreq]
#         best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
#         for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
#             tx_codebook = tx_codebook_ls[ifreq][itx]
#             tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
#                                             return_elem_gain=True)
#             rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
#                                             return_elem_gain=True)
#             for iwtx, wtx in enumerate(tx_codebook):
#                 tx_bf_ls, rx_bf_ls = [], []
#                 for ipath in range(len(aod_phi)):
#                     rx_sv = rx_svi[ipath]
#                     wrx = np.conj(rx_sv)
#                     wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
#                     rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
#                     rx_bf_ls.append(rx_bf)
#                     tx_sv = tx_svi[ipath] 
#                     tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
#                     tx_bf_ls.append(tx_bf)
#                 pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
#                 pl_min = np.min(pl_bf)
#                 pl_lin = 10**(-0.1*(pl_bf-pl_min))
#                 pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
#                 temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
#                 if temp_snr > best_snr:
#                     best_snr = temp_snr
#                     best_sect = itx
#                     best_bf_direct = iwtx
#         best_snr_freqs_ls[ifreq].append(best_snr)
#         # print(ifreq, best_snr, best_sect, best_bf_direct)

# ni = len(best_snr_freqs_ls[0])
# p = np.arange(ni)/ni
# label_ls = ['6GHz', '12GHz', '18GHz', '24GHz']
# for i in range(4):
#     plt.plot(np.sort(best_snr_freqs_ls[i]), p, label=label_ls[i])

# plt.grid(True)
# plt.xlim(-20,45)
# plt.xlabel('SNR')
# plt.ylabel('CDF')
# plt.legend()
# plt.savefig('cdf-train.png', dpi = 600)
# plt.show()


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

plt.scatter(best_snr_lower_freq_beam, best_snr_high_freq, s=9)
plt.grid(True)
plt.title('24GHz SNR')
plt.xlabel('SNR - beam select by 6GHz')
plt.ylabel('SNR - beam select by 24GHz')
plt.legend()
plt.savefig('scatter-train.png', dpi = 600)
plt.show()



# for _ in range(1):
    
#     link_idx = np.random.choice(I)
#     while chan_list[link_idx].link_state == LinkState.no_link:
#         link_idx = np.random.choice(I)
    
#     # link_idx = 68486
#     dvec = test_data['dvec'][link_idx]
#     print(dvec)
#     d3d = np.maximum(np.sqrt(np.sum(dvec**2)), 1)
#     # link_idx = 24695
#     # link_idx = 88361
#     # link_idx = 10021
#     print(link_idx, ' -----------------------------------')
#     chan = chan_list[link_idx]
#     # aod_theta = 90 - chan.ang[:,aod_theta_ind]
#     aod_theta = chan.ang[:,aod_theta_ind]
#     aod_phi = chan.ang[:,aod_phi_ind]
#     # aoa_theta = 90 - chan.ang[:,aoa_theta_ind]
#     aoa_theta = chan.ang[:,aoa_theta_ind]
#     aoa_phi = chan.ang[:,aoa_phi_ind]
#     pl_min = large_pl
#     print('aod_theta', aod_theta)
#     print('aod_phi', aod_phi)
#     # Freq K
#     # ifreq = 0 # lowest freq
#     snr = []
#     for ifreq in range(len(fc_ls)):
#         snr_ifreq = []
#     # for ifreq in range(1):
#         rx_arr = arr_ue_ls[ifreq]
#         for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
#             tx_codebook = tx_codebook_ls[ifreq][itx]
#             tx_svi, _ = tx_arr.sv(aod_phi, aod_theta,\
#                                             return_elem_gain=True)
#             rx_svi, _ = rx_arr.sv(aoa_phi, aoa_theta,\
#                                             return_elem_gain=True)
#             for iwtx, wtx in enumerate(tx_codebook):
#                 tx_bf_ls, rx_bf_ls = [], []
#                 for ipath in range(len(aod_phi)):
#                     rx_sv = rx_svi[ipath]
#                     wrx = np.conj(rx_sv)
#                     wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
#                     rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
#                     rx_bf_ls.append(rx_bf)
#                     tx_sv = tx_svi[ipath] 
#                     tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
#                     tx_bf_ls.append(tx_bf)
#                 pl_bf = chan.pl_ls[ifreq] - np.array(tx_bf_ls) - np.array(rx_bf_ls)
#                 pl_min = np.min(pl_bf)
#                 pl_lin = 10**(-0.1*(pl_bf-pl_min))
#                 pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
#                 temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
#                 snr_ifreq.append(temp_snr)
#         snr_ifreq = snr_ifreq[16:] + snr_ifreq[:16]
#         snr.append(snr_ifreq)

# min_snr, max_snr = min(snr), max(snr)
# phi_ls = np.linspace(-165, 180, 24, dtype = int)
# c = ['b', 'y', 'g', 'r']
# label_ls = ['6GHz', '12GHz', '18GHz', '24GHz']
# for i in range(4):
#     plt.plot(phi_ls, snr[i],'o-', linewidth=0.5, markersize=2,  label=label_ls[i], c=c[i])
#     plt.axhline(y=np.max(snr[i]), linestyle = '--', linewidth=1, c=c[i])
#     # plt.plot([phi_ls[best_bf]] * 2, [min_snr, max_snr], c = 'black', linewidth=0.5)
    
# # plt.grid(True)
# plt.xlabel('Beam (Angles)')
# plt.ylabel('SNR')
# # plt.title('Link - ' + str(link_idx))
# plt.title('Generated Link - ' + str(int(d3d)) + ' meters')
# plt.legend(loc = 'lower right')
# # plt.savefig('single-snr-link'+str(link_idx) + '.png', dpi = 600)
# plt.savefig('single-snr-link-Generated-'+str(int(d3d))+ '.png', dpi = 600)
# plt.show()


"""
for _ in range(10):
    
    link_idx = np.random.choice(I)
    while chan_list[link_idx].link_state == LinkState.no_link:
        link_idx = np.random.choice(I)
    
    # link_idx = 68486
    # link_idx = 24695
    # link_idx = 88361
    link_idx = 10021
    print(link_idx, ' -----------------------------------')
    chan = chan_list[link_idx]
    # aod_theta = 90 - chan.ang[:,aod_theta_ind]
    aod_theta = chan.ang[:,aod_theta_ind]
    aod_phi = chan.ang[:,aod_phi_ind]
    # aoa_theta = 90 - chan.ang[:,aoa_theta_ind]
    aoa_theta = chan.ang[:,aoa_theta_ind]
    aoa_phi = chan.ang[:,aoa_phi_ind]
    pl_min = large_pl
    print('aod_theta', aod_theta)
    print('aod_phi', aod_phi)
    # Freq K
    # ifreq = 0 # lowest freq
    for ifreq in range(len(fc_ls)):
        best_snr = []
    # for ifreq in range(1):
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
        
        print(ifreq, best_snr, best_sect, best_bf_direct)

"""

"""
for ifreq in range(len(fc_ls)):
    rx_arr = arr_ue_ls[ifreq]
    best_snr, best_sect, best_bf_direct = -float('inf'), 0, 0
    for itx, tx_arr in enumerate(arr_gnb_sects_ls[ifreq]):
        tx_svi, tx_elem_gaini = tx_arr.sv(aod_phi, aod_theta,\
                                        return_elem_gain=True)
        rx_svi, rx_elem_gaini = rx_arr.sv(aoa_phi, aoa_theta,\
                                        return_elem_gain=True)
        # tx_svi => npath x nelement
        for ibf_direct in range(len(tx_svi)):
            tx_sv = tx_svi[ibf_direct] # SECT-itx's BFDirec-ibf_direct
            rx_sv = rx_svi[ibf_direct]
            wtx = np.conj(tx_sv)
            wtx = wtx / np.sqrt(np.sum(np.abs(wtx)**2))
            wrx = np.conj(rx_sv)
            wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
            
            # Compute the gain with both the element and BF gain        
            tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx.T)))
            rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx.T)))
            bf_gain = tx_bf + rx_bf
            print(bf_gain)
            pl_bf = chan.pl_ls[ifreq] - tx_bf - rx_bf
            pl_min = np.min(pl_bf)
            pl_lin = 10**(-0.1*(pl_bf-pl_min))
            pl_eff = pl_min-10*np.log10(np.sum(pl_lin))
            temp_snr = tx_pow - pl_eff - kT - nf - 10*np.log10(bw_ls[ifreq])
            if temp_snr > best_snr:
                best_snr = temp_snr
                best_sect = itx
                best_bf_direct = ibf_direct
    print(ifreq, best_snr, best_sect, best_bf_direct)
"""
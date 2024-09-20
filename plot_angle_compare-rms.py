# -*- coding: utf-8 -*-
"""
Compute RMS angular spread for different freqs

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
Load the pre-trained model
"""
# Construct and load the channel model object
mod_name = 'Beijing'
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(model_name, cfg, 'local')

nfreq = 4
"""
Generate chan_list for test data
"""
if_real, if_generated, if_plot = False, False, True
if if_real:
    # Using the ray tracing data (real path data)
    chan_list, ls = data_to_mpchan(test_data, cfg) 
    dvec = test_data['dvec']
    fspl_ls = test_data['fspl_ls']
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
    rms_aoa_phi_ls, rms_aoa_theta_ls, rms_aod_phi_ls, rms_aod_theta_ls = [], [], [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            rms_aoa_phi_ls.append([0] * nfreq)
            rms_aoa_theta_ls.append([0] * nfreq)
            rms_aod_phi_ls.append([0] * nfreq)
            rms_aod_theta_ls.append([0] * nfreq)
            continue
        chan = chan_list[link_idx]

        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        
        rms_aoa_phi, rms_aoa_theta, rms_aod_phi, rms_aod_theta = [], [], [], []
        for ifreq in range(nfreq):
            npath = np.sum(chan.pl_ls[ifreq,:] <= fspl_ls[link_idx][ifreq] + 70)
            if npath > 0:
                pl_consider = chan.pl_ls[ifreq,:npath]
                pl_min = np.min(pl_consider)
                w = 10**(-0.1*(pl_consider-pl_min))
                w = w / np.sum(w)

                _mean = w.dot(chan.ang[:npath,0])
                rms_aoa_phi.append(np.sqrt(w.dot((chan.ang[:npath,0]-_mean)**2)))
                _mean = w.dot(chan.ang[:npath,1])
                rms_aoa_theta.append(np.sqrt(w.dot((chan.ang[:npath,1]-_mean)**2)))
                _mean = w.dot(chan.ang[:npath,2])
                rms_aod_phi.append(np.sqrt(w.dot((chan.ang[:npath,2]-_mean)**2)))
                _mean = w.dot(chan.ang[:npath,3])
                rms_aod_theta.append(np.sqrt(w.dot((chan.ang[:npath,3]-_mean)**2)))
            else:
                rms_aoa_phi.append(0)
                rms_aoa_theta.append(0)
                rms_aod_phi.append(0)
                rms_aod_theta.append(0)
        rms_aoa_phi_ls.append(rms_aoa_phi)
        rms_aoa_theta_ls.append(rms_aoa_theta)
        rms_aod_phi_ls.append(rms_aod_phi)
        rms_aod_theta_ls.append(rms_aod_theta)
    with open('rms_angular_spread_real.pkl', 'wb') as file:
        pickle.dump([rms_aoa_phi_ls, rms_aoa_theta_ls, rms_aod_phi_ls,rms_aod_theta_ls], file)



if if_generated:
    # Using the generated data (real path data)
    ls = test_data['link_state']
    fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
    fspl_ls = test_data['fspl_ls'].transpose(1,0)
    chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)
    fspl_ls = test_data['fspl_ls']
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
    rms_aoa_phi_ls, rms_aoa_theta_ls, rms_aod_phi_ls, rms_aod_theta_ls = [], [], [], []
    for link_idx in I:
        if chan_list[link_idx].link_state == LinkState.no_link:
            rms_aoa_phi_ls.append([0] * nfreq)
            rms_aoa_theta_ls.append([0] * nfreq)
            rms_aod_phi_ls.append([0] * nfreq)
            rms_aod_theta_ls.append([0] * nfreq)
            continue
        chan = chan_list[link_idx]

        aod_theta = chan.ang[:,aod_theta_ind]
        aod_phi = chan.ang[:,aod_phi_ind]
        aoa_theta = chan.ang[:,aoa_theta_ind]
        aoa_phi = chan.ang[:,aoa_phi_ind]
        pl_min = large_pl
        
        rms_aoa_phi, rms_aoa_theta, rms_aod_phi, rms_aod_theta = [], [], [], []
        for ifreq in range(nfreq):
            npath = np.sum(chan.pl_ls[ifreq,:] <= fspl_ls[link_idx][ifreq] + 70)
            if npath > 0:
                pl_consider = chan.pl_ls[ifreq,:npath]
                pl_min = np.min(pl_consider)
                w = 10**(-0.1*(pl_consider-pl_min))
                w = w / np.sum(w)

                _mean = w.dot(chan.ang[:npath,0])
                rms_aoa_phi.append(np.sqrt(w.dot((chan.ang[:npath,0]-_mean)**2)))
                _mean = w.dot(chan.ang[:npath,1])
                rms_aoa_theta.append(np.sqrt(w.dot((chan.ang[:npath,1]-_mean)**2)))
                _mean = w.dot(chan.ang[:npath,2])
                rms_aod_phi.append(np.sqrt(w.dot((chan.ang[:npath,2]-_mean)**2)))
                _mean = w.dot(chan.ang[:npath,3])
                rms_aod_theta.append(np.sqrt(w.dot((chan.ang[:npath,3]-_mean)**2)))
            else:
                rms_aoa_phi.append(0)
                rms_aoa_theta.append(0)
                rms_aod_phi.append(0)
                rms_aod_theta.append(0)
        rms_aoa_phi_ls.append(rms_aoa_phi)
        rms_aoa_theta_ls.append(rms_aoa_theta)
        rms_aod_phi_ls.append(rms_aod_phi)
        rms_aod_theta_ls.append(rms_aod_theta)
        with open('rms_angular_spread_train.pkl', 'wb') as file:
            pickle.dump([rms_aoa_phi_ls, rms_aoa_theta_ls, rms_aod_phi_ls,rms_aod_theta_ls], file)
    



if if_plot:
    rms_aoa_phi_ls_train, rms_aoa_theta_ls_train, rms_aod_phi_ls_train, rms_aod_theta_ls_train = [], [], [], []
    rms_aoa_phi_ls_real, rms_aoa_theta_ls_real, rms_aod_phi_ls_real, rms_aod_theta_ls_real = [], [], [], []
    with open('rms_angular_spread_train.pkl', 'rb') as file:
        rms_aoa_phi_ls_train, rms_aoa_theta_ls_train, rms_aod_phi_ls_train, rms_aod_theta_ls_train = pickle.load(file)
    with open('rms_angular_spread_real.pkl', 'rb') as file:
        rms_aoa_phi_ls_real, rms_aoa_theta_ls_real, rms_aod_phi_ls_real, rms_aod_theta_ls_real = pickle.load(file)
        
    selected_freq = [0, 2]
    label_ls = ['6GHz-Generated', '18GHz-Generated', '6GHz-True', '18GHz-True'] 
    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1,2, figsize=(9,4))
    ni = len(rms_aoa_phi_ls_train)
    p = np.arange(ni)/ni
    ax[0].plot(np.sort(np.array(rms_aoa_phi_ls_train)[:,0]), p, '-', color='blue')
    ax[0].plot(np.sort(np.array(rms_aoa_phi_ls_train)[:,2]), p, '-', color='red')
    ni = len(rms_aoa_phi_ls_real)
    p = np.arange(ni)/ni
    ax[0].plot(np.sort(np.array(rms_aoa_phi_ls_real)[:,0]), p, '--', color='blue')
    ax[0].plot(np.sort(np.array(rms_aoa_phi_ls_real)[:,2]), p, '--', color='red')
    ax[0].grid()
    ax[0].set_title('Azimuth AoA')
    
    ni = len(rms_aod_phi_ls_train)
    p = np.arange(ni)/ni
    ax[1].plot(np.sort(np.array(rms_aod_phi_ls_train)[:,0]), p, '-', color='blue')
    ax[1].plot(np.sort(np.array(rms_aod_phi_ls_train)[:,2]), p, '-', color='red')
    ni = len(rms_aod_phi_ls_real)
    p = np.arange(ni)/ni
    ax[1].plot(np.sort(np.array(rms_aod_phi_ls_real)[:,0]), p, '--', color='blue')
    ax[1].plot(np.sort(np.array(rms_aod_phi_ls_real)[:,2]), p, '--', color='red')
    ax[1].grid()
    ax[1].set_title('Azimuth AoD')
    
    ax[0].set_ylim([0, 1.0])
    ax[1].set_ylim([0, 1.0])
    ax[0].set_xlim([0, 150])
    ax[1].set_xlim([0, 150])
    ax[0].set_ylabel('CDF')
    ax[1].set_yticklabels([])  # 删除其他子图的y轴标签
    # ax[0].set_xlabel('RMS Angular Spread')
    # ax[1].set_xlabel('RMS Angular Spread')
    fig.text(0.5, 0.015, 'RMS Angular Spread', ha='center')
    ax[1].legend(label_ls, loc='lower right')
    fig.tight_layout()
    plt.savefig('0225-6GHz-reduce70%-New/RMS-angular-spread-cdf.png', dpi = 400)
    plt.show()
    
    # selected_freq = [0, 2]
    # label_ls = ['6GHz-Generated', '18GHz-Generated', '6GHz-True', '18GHz-True'] 
    # matplotlib.rcParams.update({'font.size': 15})
    # fig, ax = plt.subplots(2,2, figsize=(9,9))
    
    # ni = len(rms_aoa_phi_ls_train)
    # p = np.arange(ni)/ni
    # ax[0][0].plot(np.sort(np.array(rms_aoa_phi_ls_train)[:,0]), p, '-', color='blue')
    # ax[0][0].plot(np.sort(np.array(rms_aoa_phi_ls_train)[:,2]), p, '-', color='red')
    # ni = len(rms_aoa_phi_ls_real)
    # p = np.arange(ni)/ni
    # ax[0][0].plot(np.sort(np.array(rms_aoa_phi_ls_real)[:,0]), p, '--', color='blue')
    # ax[0][0].plot(np.sort(np.array(rms_aoa_phi_ls_real)[:,2]), p, '--', color='red')
    # ax[0][0].grid()
    # ax[0][0].set_title('Azimuth AoA')
    
    # ni = len(rms_aoa_theta_ls_train)
    # p = np.arange(ni)/ni
    # ax[0][1].plot(np.sort(np.array(rms_aoa_theta_ls_train)[:,0]), p, '-', color='blue')
    # ax[0][1].plot(np.sort(np.array(rms_aoa_theta_ls_train)[:,2]), p, '-', color='red')
    # ni = len(rms_aoa_theta_ls_real)
    # p = np.arange(ni)/ni
    # ax[0][1].plot(np.sort(np.array(rms_aoa_theta_ls_real)[:,0]), p, '--', color='blue')
    # ax[0][1].plot(np.sort(np.array(rms_aoa_theta_ls_real)[:,2]), p, '--', color='red')
    # ax[0][1].grid()
    # ax[0][1].set_title('Elevation AoA')
    
    # ni = len(rms_aod_phi_ls_train)
    # p = np.arange(ni)/ni
    # ax[1][0].plot(np.sort(np.array(rms_aod_phi_ls_train)[:,0]), p, '-', color='blue')
    # ax[1][0].plot(np.sort(np.array(rms_aod_phi_ls_train)[:,2]), p, '-', color='red')
    # ni = len(rms_aod_phi_ls_real)
    # p = np.arange(ni)/ni
    # ax[1][0].plot(np.sort(np.array(rms_aod_phi_ls_real)[:,0]), p, '--', color='blue')
    # ax[1][0].plot(np.sort(np.array(rms_aod_phi_ls_real)[:,2]), p, '--', color='red')
    # ax[1][0].grid()
    # ax[1][0].set_title('Azimuth AoD')
    
    # ni = len(rms_aod_theta_ls_train)
    # p = np.arange(ni)/ni
    # ax[1][1].plot(np.sort(np.array(rms_aod_theta_ls_train)[:,0]), p, '-', color='blue')
    # ax[1][1].plot(np.sort(np.array(rms_aod_theta_ls_train)[:,2]), p, '-', color='red')
    # ni = len(rms_aod_theta_ls_real)
    # p = np.arange(ni)/ni
    # ax[1][1].plot(np.sort(np.array(rms_aod_theta_ls_real)[:,0]), p, '--', color='blue')
    # ax[1][1].plot(np.sort(np.array(rms_aod_theta_ls_real)[:,2]), p, '--', color='red')
    # ax[1][1].grid()
    # ax[1][1].legend(label_ls, loc='lower right')
    # ax[1][1].set_title('Elevation AoD')
    
    
    # for i in range(2):
    #     ax[i][0].set_ylim([0, 1.0])
    #     ax[i][0].set_xlim([0, 150])
    #     ax[i][1].set_ylim([0, 1.0])
    #     ax[i][1].set_xlim([0, 30])
    #     ax[i][0].set_ylabel('CDF')
    #     ax[i][1].set_yticklabels([])  # 删除其他子图的y轴标签
    #     ax[0][i].set_xticklabels([])  # 删除其他子图的y轴标签
    #     ax[1][i].set_xlabel('RMS Angular Spread')
    
            
    # # fig.text(0.5, 0.00, 'SNR Difference (dB) Between Selections at Higher Frequencies and 6GHz', ha='center')
    # fig.tight_layout()
    # # plt.savefig('0225-6GHz-reduce70%-New/snr-beam-selected-cdf.png', dpi = 400)
    # plt.show()

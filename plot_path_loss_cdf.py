"""
plot_path_loss_cdf2:  Plots the CDF of the path loss on the test data,
and compares that to the randomly generated path loss from the trained model.
"""
import os
import pickle
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
import tensorflow.keras.backend as K

# path = os.path.abspath('../..')
# if not path in sys.path:
#     sys.path.append(path)

from mmwchanmod.common.constants import LinkState
from mmwchanmod.datasets.download import load_model
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.common.constants import DataConfig
from mmwchanmod.learn.models import ChanMod

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the omni directional path loss CDF')
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='pl_cdf.png', help='plot file name')        
parser.add_argument(\
    '--ds_city',action='store',\
    default='Beijing', help='data set to load')    
parser.add_argument(\
    '--model_city',action='store',\
    default='Beijing', help='cities for the models to test')    

args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
ds_city = args.ds_city
model_city = args.model_city
plot_fn = args.plot_fn

"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

city_test = [ds_city] + model_city.split()
use_true_ls = False

"""
Find the path loss CDFs
"""
pl_omni_plot = []
ls_plot = []
leg_str = []

ntest = len(city_test)
for i, city in enumerate(city_test):
    
    if (i == 0):
        """
        For first city, use the city data
        """
        # Convert data to channel list
        chan_list, ls = data_to_mpchan(test_data, cfg)
        # leg_str.append(city + ' data')
        leg_str.append('Generated')

    else:
        """
        For subsequent cities, generate data from model
        """
        # Construct the channel model object
        K.clear_session()
        chan_mod = load_model(model_name, cfg, 'local')
        mod_name = 'Beijing'
        # Load the configuration and link classifier model
        print('Simulating model %s'%mod_name)        
        
        # Generate samples from the path
        if use_true_ls:
            ls = test_data['link_state']
        else:
            ls = None
        fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
        fspl_ls = test_data['fspl_ls'].transpose(1,0)
        chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)
            
        # leg_str.append(city + ' model') 
        leg_str.append('True') 

        
    # Compute the omni-directional path loss for each link    
    n = len(chan_list)
    pl_omni_ls = np.zeros((cfg.nfreq, n))

    for i, chan in enumerate(chan_list):
        if chan.link_state != LinkState.no_link:
            pl_omni_ls[:, i] = chan.comp_omni_path_loss()

    # Save the results    
    ls_plot.append(ls)
    pl_omni_plot.append(pl_omni_ls)
    
"""
Create the plot
"""
ntypes = len(cfg.rx_types)
nplot = len(pl_omni_plot)
matplotlib.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(2, 2, figsize=(10,15))  # 修改为2x2布局

ax = ax.flatten()  # 将 ax 转换为一维数组以简化索引


for i, rx_type in enumerate(cfg.rx_types):
    # Plot color    
    for iplot in range(nplot):
        # Find the links that match the type and are not in outage
        I = np.where((test_data['rx_type']==i) & (ls_plot[iplot] != LinkState.no_link))[0]
        # print(iplot,len(I))
        # I = np.where((test_data['rx_type']==i) & (ls_plot[1] != LinkState.no_link))[0]
        # Select color and fmt
        if (iplot == 0):
            fmt = '-'
            color = [0,0,1]
        else:
            fmt = '--'
            t = (iplot-1)/(nplot-1)
            color = [0,t,1-t]

        # Plot the omni-directional path loss                 
        ni = len(I)
        p = np.arange(ni)/ni
        for ifreq in range(cfg.nfreq):
            pl_plot = pl_omni_plot[iplot]
            ax[ifreq].plot(np.sort(pl_plot[ifreq][I]), p, fmt, color=color)

for ifreq in range(cfg.nfreq):
    ax[ifreq].set_title(str(int(cfg.freq_set[ifreq]/1e9))+' GHz')
    if ifreq == 0 or ifreq == 2:  # 适当调整以在左侧列添加 Y 轴标签
        ax[ifreq].set_ylabel('CDF')
    if ifreq == 2 or ifreq == 3:
        ax[ifreq].set_xlabel('Path loss (dB)')
    ax[ifreq].grid()
ax[3].legend(leg_str,loc='lower right')
# fig.legend(leg_str, borderaxespad=0.1, loc='lower right', bbox_to_anchor=(0, 0.05, 1, 0.25)) # bbox_to_anchor=(0, 0.15, 1, 0.85)

# Print plot
if 1:
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    fig.tight_layout()
    fig.savefig(plot_path)
    print('Figure saved to %s' % plot_path)
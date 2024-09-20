"""
plot_path_loss_cdf2:  Plots the CDF of the path loss on the test data,
and compares that to the randomly generated path loss from the trained model.
"""
import os
import pickle
import numpy as np
import seaborn as sns
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
import tensorflow.keras.backend as K
import pickle   

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
    default='pl_vs_kde.png', help='plot file name')        
parser.add_argument(\
    '--ds_city',action='store',\
    default='Beijing', help='data set to load')    
parser.add_argument(\
    '--model_city',action='store',\
    default='Beijing', help='cities for the models to test')    

parser.add_argument(\
    '--if_plot',action='store',\
    default=0, help='If plots')
parser.add_argument(\
    '--if_save',action='store',\
    default=0, help='If save the results')


args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
ds_city = args.ds_city
model_city = args.model_city
plot_fn = args.plot_fn
if_plot = int(args.if_plot)
if_save = int(args.if_save)
print(if_plot, if_save)

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

if if_save == 1:
    """
    Find the path loss CDFs
    """
    pl_omni_plot = []
    pl2_omni_plot = []
    pl3_omni_plot = []
    pl4_omni_plot = []
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
            
            leg_str.append(city + ' data')
            
            
        else:
            """
            For subsequent cities, generate data from model
            """
            # Construct the channel model object
            K.clear_session()
            chan_mod = load_model(model_name,cfg, 'local')
            
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
                
            leg_str.append(city + ' model') 

            
        # Compute the omni-directional path loss for each link    
        n = len(chan_list)
        pl_omni = np.zeros(n)
        pl2_omni = np.zeros(n)
        pl3_omni = np.zeros(n)
        pl4_omni = np.zeros(n)
        for i, chan in enumerate(chan_list):
            if chan.link_state != LinkState.no_link:
                pl_omni[i], pl2_omni[i], pl3_omni[i], pl4_omni[i]= chan.comp_omni_path_loss()

        # Save the results    
        ls_plot.append(ls)
        pl_omni_plot.append(pl_omni)
        pl2_omni_plot.append(pl2_omni)
        pl3_omni_plot.append(pl3_omni)
        pl4_omni_plot.append(pl4_omni)


    with open('array_file.pkl', 'wb') as file:
        pickle.dump([ls_plot, pl_omni_plot, pl2_omni_plot, pl3_omni_plot, pl4_omni_plot], file)
else:
    with open('array_file.pkl', 'rb') as file:
        loaded_array = pickle.load(file)
        ls_plot, pl_omni_plot, pl2_omni_plot, pl3_omni_plot, pl4_omni_plot = loaded_array[0], loaded_array[1], loaded_array[2], loaded_array[3], loaded_array[4]


if if_plot == 1:
    """
    Create the plot
    """
    ntypes = len(cfg.rx_types)
    nplot = len(pl_omni_plot)
    print(ntypes, nplot)

    matplotlib.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(2,6, figsize=(24,8))
    ax = ax.flatten()
    for i, rx_type in enumerate(cfg.rx_types):
        
  
        for iplot in range(nplot):
            
                dvec = test_data['dvec']
                d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)

                # Find the links that match the type and are not in outage
                I = np.where((test_data['rx_type']==i) & (d3d>=100)& (d3d<=200)&\
                    (ls_plot[iplot] != LinkState.no_link))[0]


                # Plot the omni-directional path loss                 
                ni = len(I)
                p = np.arange(ni)/ni            
                if iplot == 0:
                    sns.kdeplot(pl_omni_plot[iplot][I], pl2_omni_plot[iplot][I], fill = True, ax=ax[0])
                    sns.set_style("whitegrid")
                    # ax[0].set_title('Ray Tracing Samples')   
                    ax[0].set_xlabel('6GHz Path loss(dB)')
                    ax[0].set_ylabel('12GHz Path loss(dB)')
                    ax[0].set_xlim(70,200)
                    ax[0].set_ylim(95,180)
                    ax[0].grid()

                    sns.kdeplot(pl_omni_plot[iplot][I], pl3_omni_plot[iplot][I], fill = True, ax=ax[1])
                    sns.set_style("whitegrid")
                    # ax[1].set_title('Ray Tracing Samples')   
                    ax[1].set_xlabel('6GHz Path loss(dB)')
                    ax[1].set_ylabel('18GHz Path loss(dB)')
                    ax[1].set_xlim(70,200)
                    ax[1].set_ylim(95,180)
                    ax[1].grid()

                    sns.kdeplot(pl_omni_plot[iplot][I], pl4_omni_plot[iplot][I], fill = True, ax=ax[2])
                    sns.set_style("whitegrid")
                    # ax[2].set_title('Ray Tracing Samples')   
                    ax[2].set_xlabel('6GHz Path loss(dB)')
                    ax[2].set_ylabel('24GHz Path loss(dB)')
                    ax[2].set_xlim(70,200)
                    ax[2].set_ylim(95,180)
                    ax[2].grid()

                    sns.kdeplot(pl2_omni_plot[iplot][I], pl3_omni_plot[iplot][I], fill = True, ax=ax[3])
                    sns.set_style("whitegrid")
                    # ax[3].set_title('Ray Tracing Samples')   
                    ax[3].set_xlabel('12GHz Path loss(dB)')
                    ax[3].set_ylabel('18GHz Path loss(dB)')
                    ax[3].set_xlim(70,200)
                    ax[3].set_ylim(95,180)
                    ax[3].grid()

                    sns.kdeplot(pl2_omni_plot[iplot][I], pl4_omni_plot[iplot][I], fill = True, ax=ax[4])
                    sns.set_style("whitegrid")
                    # ax[4].set_title('Ray Tracing Samples')   
                    ax[4].set_xlabel('12GHz Path loss(dB)')
                    ax[4].set_ylabel('24GHz Path loss(dB)')
                    ax[4].set_xlim(70,200)
                    ax[4].set_ylim(95,180)
                    ax[4].grid()

                    sns.kdeplot(pl3_omni_plot[iplot][I], pl4_omni_plot[iplot][I], fill = True, ax=ax[5])
                    sns.set_style("whitegrid")
                    # ax[5].set_title('Ray Tracing Samples')   
                    ax[5].set_xlabel('18GHz Path loss(dB)')
                    ax[5].set_ylabel('24GHz Path loss(dB)')
                    ax[5].set_xlim(70,200)
                    ax[5].set_ylim(95,180)
                    ax[5].grid()

                else:

                    sns.kdeplot(pl_omni_plot[iplot][I], pl2_omni_plot[iplot][I], fill = True, ax=ax[6])
                    sns.set_style("whitegrid")
                    # ax[6].set_title('Generated Links')   
                    ax[6].set_xlabel('6GHz Path loss(dB)')
                    ax[6].set_ylabel('12GHz Path loss(dB)')
                    ax[6].set_xlim(70,200)
                    ax[6].set_ylim(95,180)
                    ax[6].grid()

                    sns.kdeplot(pl_omni_plot[iplot][I], pl3_omni_plot[iplot][I], fill = True, ax=ax[7])
                    sns.set_style("whitegrid")
                    # ax[7].set_title('Generated Links')   
                    ax[7].set_xlabel('6GHz Path loss(dB)')
                    ax[7].set_ylabel('18GHz Path loss(dB)')
                    ax[7].set_xlim(70,200)
                    ax[7].set_ylim(95,180)
                    ax[7].grid()

                    sns.kdeplot(pl_omni_plot[iplot][I], pl4_omni_plot[iplot][I], fill = True, ax=ax[8])
                    sns.set_style("whitegrid")
                    # ax[8].set_title('Generated Links')   
                    ax[8].set_xlabel('6GHz Path loss(dB)')
                    ax[8].set_ylabel('24GHz Path loss(dB)')
                    ax[8].set_xlim(70,200)
                    ax[8].set_ylim(95,180)
                    ax[8].grid()

                    sns.kdeplot(pl2_omni_plot[iplot][I], pl3_omni_plot[iplot][I], fill = True, ax=ax[9])
                    sns.set_style("whitegrid")
                    # ax[9].set_title('Generated Links')   
                    ax[9].set_xlabel('12GHz Path loss(dB)')
                    ax[9].set_ylabel('18GHz Path loss(dB)')
                    ax[9].set_xlim(70,200)
                    ax[9].set_ylim(95,180)
                    ax[9].grid()

                    sns.kdeplot(pl2_omni_plot[iplot][I], pl4_omni_plot[iplot][I], fill = True, ax=ax[10])
                    sns.set_style("whitegrid")
                    # ax[10].set_title('Generated Links')   
                    ax[10].set_xlabel('12GHz Path loss(dB)')
                    ax[10].set_ylabel('24GHz Path loss(dB)')
                    ax[10].set_xlim(70,200)
                    ax[10].set_ylim(95,180)
                    ax[10].grid()

                    sns.kdeplot(pl3_omni_plot[iplot][I], pl4_omni_plot[iplot][I], fill = True, ax=ax[11])
                    sns.set_style("whitegrid")
                    # ax[11].set_title('Generated Links')   
                    ax[11].set_xlabel('18GHz Path loss(dB)')
                    ax[11].set_ylabel('24GHz Path loss(dB)')
                    ax[11].set_xlim(70,200)
                    ax[11].set_ylim(95,180)
                    ax[11].grid()



            # plt.scatter(d3d[I], pl2_omni_plot[iplot][I], s=9)
                  

    # plt.legend(['data_2.3GHz','data_28GHz', 'model_2.3GHz','model_28GHz'],loc='lower right',ncol=2)


    # Print plot
    if 1:
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
            print('Created directory %s' % plot_dir)
        plot_path = os.path.join(plot_dir, plot_fn)
        fig.text(0.5, 0.97, 'Ray Tracing Samples', ha='center')
        fig.text(0.5, 0.49, 'Generated Links', ha='center')
        fig.tight_layout()
        plt.savefig(plot_path)
        print('Figure saved to %s' % plot_path)
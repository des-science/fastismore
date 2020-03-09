#!/usr/bin/env python
# coding: utf-8

# Number of points to burn-in

# Parameters to plot
params2plot = [
     'cosmological_parameters--omega_m',
#     'cosmological_parameters--omega_c',
#     'cosmological_parameters--omch2',
#     'cosmological_parameters--ombh2',
#     'cosmological_parameters--w0_fld',
#     'cosmological_parameters--log_cs2',
#     'cosmological_parameters--h0',
#     'cosmological_parameters--omega_b',
#     'cosmological_parameters--n_s',
#     'cosmological_parameters--a_s',
#     'cosmological_parameters--omnuh2',
#     'cosmological_parameters--sigma_8',
     'cosmological_parameters--s8',
#     'cosmological_parameters--w',
#     'cosmological_parameters--wa',
#     'intrinsic_alignment_parameters--a',
#     'intrinsic_alignment_parameters--alpha',
#     'bin_bias--b1',
#     'bin_bias--b2',
#     'bin_bias--b3',
#     'bin_bias--b4',
#     'bin_bias--b5',
#     'shear_calibration_parameters--m1',
#     'shear_calibration_parameters--m2',
#     'shear_calibration_parameters--m3',
#     'shear_calibration_parameters--m4',
#     'lens_photoz_errors--bias_1',
#     'lens_photoz_errors--bias_2',
#     'lens_photoz_errors--bias_3',
#     'lens_photoz_errors--bias_4',
#     'lens_photoz_errors--bias_5',
#     'wl_photoz_errors--bias_1',
#     'wl_photoz_errors--bias_2',
#     'wl_photoz_errors--bias_3',
#     'wl_photoz_errors--bias_4',
]

import numpy as np
import re
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import itertools as itt
from chainconsumer import ChainConsumer
import sys

import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)


# -------------------- BASIC DEFINITIONS ---------------------

label_dict = {'cosmological_parameters--w0_fld':  r'w_{GDM}',
              'cosmological_parameters--cs2_fld': r'c_s^2',
              'cosmological_parameters--log_cs2': r'log(c_s^2)',
              'cosmological_parameters--omega_m': r'\Omega_m',
              'cosmological_parameters--omega_c': r'\Omega_c',
              'cosmological_parameters--ommh2': r'\Omega_m h^2',
              'cosmological_parameters--ombh2': r'\Omega_b h^2',
              'cosmological_parameters--omch2': r'\Omega_c h^2',
              'cosmological_parameters--h0':      r'h',
              'cosmological_parameters--omega_b': r'\Omega_b',
              'cosmological_parameters--n_s':     r'n_s',
              'cosmological_parameters--a_s':     r'A_s',
              'cosmological_parameters--omnuh2':  r'\Omega_{\nu}',
              'cosmological_parameters--sigma_8': r'\sigma_8',
              'cosmological_parameters--s8': r'S_8',
              'cosmological_parameters--w': r'w',
              'cosmological_parameters--wa': r'w_a',
              'intrinsic_alignment_parameters--a': r'A_{IA}',
              'intrinsic_alignment_parameters--alpha': r'\alpha_{IA}',
              'bin_bias--b1': 'b_1',
              'bin_bias--b2': 'b_2',
              'bin_bias--b3': 'b_3',
              'bin_bias--b4': 'b_4',
              'bin_bias--b5': 'b_5',
              'shear_calibration_parameters--m1': 'm_1',
              'shear_calibration_parameters--m2': 'm_2',
              'shear_calibration_parameters--m3': 'm_3',
              'shear_calibration_parameters--m4': 'm_4',
              'lens_photoz_errors--bias_1': 'z^l_1',
              'lens_photoz_errors--bias_2': 'z^l_2',
              'lens_photoz_errors--bias_3': 'z^l_3',
              'lens_photoz_errors--bias_4': 'z^l_4',
              'lens_photoz_errors--bias_5': 'z^l_5',
              'wl_photoz_errors--bias_1': 'z^s_1',
              'wl_photoz_errors--bias_2': 'z^s_2',
              'wl_photoz_errors--bias_3': 'z^s_3',
              'wl_photoz_errors--bias_4': 'z^s_4',
              }

def load_chain(filename, burn=0):
    data = []
    with open(filename) as f:
        labels = np.array(f.readline()[1:-1].lower().split())
        mask = ["data_vector" not in l for l in labels]
        for line in f.readlines():
            if '#' in line:
                continue
            else:
                data.append(np.array(line.split(), dtype=np.double)[mask])
    data = {labels[mask][i]: col for i,col in enumerate(np.array(data)[burn:,:].T)}
    return data

get_label = np.vectorize(lambda label: label_dict[label] if label in label_dict.keys() else label)

def on_params(arr, params2plot):
    return np.array([arr[l] for l in params2plot if l in arr.keys()]).T

def add_S8(data):
    data['cosmological_parameters--s8'] = data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.5
    return data

# Computes ommh2, ombh2, omch2, omega_c
def add_omxh2(data):
    data['cosmological_parameters--ommh2'] = data['cosmological_parameters--omega_m']*data['cosmological_parameters--h0']**2
    data['cosmological_parameters--ombh2'] = data['cosmological_parameters--omega_b']*data['cosmological_parameters--h0']**2
    data['cosmological_parameters--omch2'] = data['cosmological_parameters--ommh2'] - data['cosmological_parameters--ombh2']
    data['cosmological_parameters--omega_c'] = data['cosmological_parameters--omega_m'] - data['cosmological_parameters--omega_b']
    return data

def load_baseline_chain(baseline_filename, params2plot, burn=0):
    """load baseline chain, add derived parameters, return (baseline_data, chainconsumer object)"""
    c = ChainConsumer()
    
    data = load_chain(baseline_filename, burn=burn)
    data = add_S8(data)
    #data = add_omxh2(data)
    
    # Add baseline chain
    c.add_chain(on_params(data, params2plot), weights=data['weight'] if 'weight' in data.keys() else None,
                parameters=['$'+l+'$' for l in get_label(params2plot)], name='Base')
    return data, c
    
def get_dloglike_stats(ISdata):
    """compute weighted average and rms of loglikelihood difference. Should be <~ O(1).
    returns (dloglike, rms_dloglike)"""
    dloglike = -np.average(ISdata['new_like'] - ISdata['old_like'], weights=ISdata['old_weight'])
    rmsdloglike = np.average((ISdata['new_like'] - ISdata['old_like'])**2, weights=ISdata['old_weight'])**0.5
    
    return dloglike, rmsdloglike
    
    
def add_IS_chain(cc_base, filename, params2plot, baseline_data, burn=0, label='IS', append_stats=True):
    """load and add IS chain to current chainconsumer object.
    Return (cc, ISdata)"""
    ISdata = load_chain(filename, burn=burn)
    dloglike, rmsdloglike = get_dloglike_stats(ISdata)
    if append_stats:
        label = label+'({:.2g})'.format(dloglike)

    cc_base.add_chain(on_params(baseline_data, params2plot), weights=ISdata['weight'],
                parameters=['$'+l+'$' for l in get_label(params2plot)], name=label)
    return cc_base, ISdata
    
def set_default_cc_config(cc):
    """set chainconsumer configuration. Call after adding all chains."""
    cc.configure(linestyles="-", linewidths=1.0,
            shade=False, shade_alpha=0.5, sigmas=[1,2], kde=False,
#            colors=['blue', 'red'],
            label_font_size=20, tick_font_size=20, legend_color_text=True)
    return cc       
    
def plot_weights(ISdata, label='IS', ax=None, plotbaseline=True, stats=True):
    if ax is None:
        f, ax = plt.subplots(figsize=(10,2))
#    f = plt.figure(figsize=(10,2))
    if plotbaseline:
        ax.plot(ISdata['old_weight'], label='Baseline', zorder=2)
    ax.plot(ISdata['weight'], label=label, alpha=0.5)
    ax.set_ylabel('weight', fontsize=20)
    ax.set_xlabel('Sample', fontsize=20)
    if stats:
        ax.text(0.01, 0.95, '({:.2g}, {:.2g})'.format(*get_dloglike_stats(ISdata)),
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, fontsize=16)
    ax.legend(loc=2, fontsize=20)
    return ax
    
## ---------------------------- PLOT ----------------------------
#
#c = ChainConsumer()
#
#data = load_chain(chain_filename, burn=burn)
#data = add_S8(data)
##data = add_omxh2(data)
#
## Add baseline chain
#c.add_chain(on_params(data, params2plot), weights=data['weight'] if 'weight' in data.keys() else None,
#            parameters=['$'+l+'$' for l in get_label(params2plot)], name='Base')
#
## Add importance sampled chains with IS weight
#for filename in is_filelist:
#    # IS weights
#    #weights = np.e**(np.loadtxt(filename)[burn:]+data['prior']-data['post'])
#    #if 'weight' in data.keys():
#    #    weights *= data['weight']
#
#    weights = load_chain(filename)['weight'][burn:]
#
#    c.add_chain(on_params(data, params2plot), weights=weights,
#                parameters=['$'+l+'$' for l in get_label(params2plot)], name='IS')
#
## Some plot configurations
#c.configure(linestyles="-", linewidths=1.0,
#            shade=False, shade_alpha=0.5, sigmas=[1,2], kde=False,
##            colors=['blue', 'red'],
#            label_font_size=20, tick_font_size=20, legend_color_text=True)
#
#fig = c.plotter.plot()
#fig.set_size_inches(4.5 + fig.get_size_inches())
#fig.savefig(output_filename)
#
##--------
    
if __name__ == "__main__":
    print('RUNNING AS MAIN')
    burn = 0
    chain_filename = sys.argv[1]
    is_filelist = sys.argv[2:-1]
    output_filename = sys.argv[-1]
    
    N_IS = len(is_filelist)
    
    data_bl, cc = load_baseline_chain(chain_filename, params2plot, burn=burn)
    f, axes = plt.subplots(N_IS, figsize=(10, 2*N_IS))
    for i,filename in enumerate(is_filelist):
        cc, ISdata = add_IS_chain(cc, filename, params2plot, baseline_data=data_bl, burn=burn, label='IS{}'.format(i))
        ax = axes[i] if N_IS>1 else axes
        ax = plot_weights(ISdata, label='IS{}'.format(i), plotbaseline=True, ax=ax, stats=True)
 
    f.savefig(output_filename.split('.')[0] + '_weights.png', bbox_inches='tight')
    
    #plot posteriors    
    cc = set_default_cc_config(cc)
        
    fig = cc.plotter.plot()
    fig.set_size_inches(4.5 + fig.get_size_inches())
    fig.savefig(output_filename, bbox_inches='tight')

#!/usr/bin/env python
# coding: utf-8

# Number of points to burn-in
burn = 0

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
import matplotlib.pyplot as plot
import itertools as itt
from chainconsumer import ChainConsumer
import sys

import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

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
    fiducial = {}
    prior_range = {}

    with open(filename) as f:
        labels = np.array(f.readline()[1:-1].lower().split())

    return {labels[i]: l for i, l in enumerate(np.loadtxt(filename)[burn:,:].T)}

get_label = np.vectorize(lambda label: label_dict[label] if label in label_dict.keys() else label)

def on_params(arr, params2plot):
    return np.array([arr[l] for l in params2plot]).T

def add_S8(data):
    data['cosmological_parameters--s8'] = data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.5
    return data

def add_omxh2(data):
    data['cosmological_parameters--ommh2'] = data['cosmological_parameters--omega_m']*data['cosmological_parameters--h0']**2
    data['cosmological_parameters--ombh2'] = data['cosmological_parameters--omega_b']*data['cosmological_parameters--h0']**2
    data['cosmological_parameters--omch2'] = data['cosmological_parameters--ommh2'] - data['cosmological_parameters--ombh2']
    data['cosmological_parameters--omega_c'] = data['cosmological_parameters--omega_m'] - data['cosmological_parameters--omega_b']
    return data

c = ChainConsumer()

data = load_chain(sys.argv[1], burn=burn)
data = add_S8(data)
data = add_omxh2(data)

c.add_chain(on_params(data, params2plot), weights=data['weight'] if 'weight' in data.keys() else None,
            parameters=['$'+l+'$' for l in get_label(params2plot)], name='Base')

for filename in sys.argv[2:-1]:
    weights = np.e**(np.loadtxt(filename)[burn:]-data['post'])
    if 'weight' in data.keys():
        weights *= data['weight']

    c.add_chain(on_params(data, params2plot), weights=weights,
                parameters=['$'+l+'$' for l in get_label(params2plot)], name='IS')



c.configure(linestyles="-", linewidths=1.0,
            shade=False, shade_alpha=0.5, sigmas=[1,2], kde=False,
#            colors=['blue', 'red'],
            label_font_size=20, tick_font_size=20, legend_color_text=True)

fig = c.plotter.plot()
fig.set_size_inches(4.5 + fig.get_size_inches())
fig.savefig(sys.argv[-1])


# ## Plot weights

#if(len(sys.argv) > 3):
#    f = plot.figure(figsize=(10,2))
#    plot.plot(data['weight'], label='Baseline', zorder=2)
#    plot.title('weights', fontsize=20)
#    plot.legend(loc=2, fontsize=20)
#    plot.savefig(sys.argv[3])


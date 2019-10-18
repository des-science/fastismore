#!/usr/bin/env python
# author: Otavio Alves
#
# description: This code computes -1/2*chi2 for a data vector given a chain with data_vector--2pt_theory_### columns
#
# input: data_vector.fits: cosmosis-like data vector
#        chain.txt: chain with data_vector--2pt_theory_### columns to compute chi2
#
# output: output.txt: -1/2*chi2 (log-likelihood) for each point in chain.txt
#

import sys
assert len(sys.argv) == 4, "Should have 3 arguments: data_vector.fits chain.txt output.txt"

import numpy as np
import pandas as pd
from astropy.io import fits
import twopoint # from cosmosis/cosmosis-standard-library/2pt/
import configparser

data_sets = ['xip', 'xim', 'gammat', 'wtheta']

# ---------- Do scale cuts -------------

with open(sys.argv[2]) as f:
    labels = np.array(f.readline()[1:-1].lower().split())

    print('Looking for PARAMS_INI to get scale cuts...')
    line = f.readline()
    while(line != '' and line.strip() != '## START_OF_PARAMS_INI'):
        line = f.readline()

    inifile = []
    line = f.readline()
    while(line != '' and line.strip() != '## END_OF_PARAMS_INI'):
        inifile.append(line[3:].strip())
        line = f.readline()

assert len(inifile) > 0

inifile = '\n'.join(inifile)

print('Found PARAMS_INI')

values = configparser.ConfigParser(strict=False)
values.read_string(inifile)
scale_cuts = {}

data_vector = twopoint.TwoPointFile.from_fits(sys.argv[1], 'covmat')
for name in data_sets: 
    s = data_vector.get_spectrum(name)
    for b1, b2 in s.bin_pairs:
        option_name = "angle_range_{}_{}_{}".format(name, b1, b2)
        r = [float(e) for e in values.get('2pt_like', option_name).split()]
        scale_cuts[(name, b1, b2)] = r

data_vector.mask_scales(scale_cuts)
spectra = [data_vector.get_spectrum(s) for s in data_sets]

e = []
for s in spectra:
    e.append(np.concatenate([s.get_pair(*p)[1] for p in s.get_bin_pairs()]))

data = np.concatenate(e)

# -------------- Scale cuts done --------------

prec = np.linalg.inv(data_vector.covmat)

print('Evaluating likelihoods...')

with open(sys.argv[2]) as f:
    mask = np.array(['data_vector--2pt_theory_' in l for l in labels])

    line = f.readline()
    while('#' in line): 
        line = f.readline()

    with open(sys.argv[3], 'w+') as output:
        while(line != ''):
            theory = np.array(line.split(), dtype=np.float64)[mask]
            d = data - theory 
            loglike = -np.einsum('i,ij,j', d, prec, d)/2
            output.write(str(loglike)+'\r\n')

            line = f.readline()

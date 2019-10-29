#!/usr/bin/env python
# author: Otavio Alves
#
# description: This code computes importance weights for a data vector given a chain with data_vector--2pt_theory_### columns
#
# input: data_vector.fits: cosmosis-like data vector
#        chain.txt: chain with data_vector--2pt_theory_### columns to compute chi2
#
# output: output.txt: -1/2*chi2 (log-likelihood) and weight for each point in chain.txt
#

import sys
assert len(sys.argv) == 4, "Should have 3 arguments: chain.txt data_vector.fits output.txt"

import numpy as np
import pandas as pd
from astropy.io import fits
import twopoint # from cosmosis/cosmosis-standard-library/2pt/
import configparser

data_sets = ['xip', 'xim', 'gammat', 'wtheta']

chain_filename = sys.argv[1]
data_vector_filename = sys.argv[2]
output_filename = sys.argv[3]

# ---------- Do scale cuts -------------

with open(chain_filename) as f:
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

data_vector = twopoint.TwoPointFile.from_fits(data_vector_filename, 'covmat')
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

#like_i = labels.index('like')
#weight_i = labels.index('weight') if 'weight' in labels else -1
#theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])

like_i = np.where(labels == 'like')
weight_i = np.where(labels == 'weight') if 'weight' in labels else -1
theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])

total_is = 0
norm_fact = 0

print('Evaluating likelihoods...')

with open(chain_filename) as f:

    with open(output_filename, 'w+') as output:
        output.write('#new_like\tweight\r\n')
        output.write('#\r\n')
        output.write('# Importance sampling weights\r\n')
        output.write('# Chain: {}\r\n'.format(chain_filename))
        output.write('# Data vector: {}\r\n'.format(data_vector_filename))
        output.write('# Data vector size: {}\r\n'.format(np.sum(theory_i)))
        output.write('# Previous weights {}.\r\n'.format('were found and incorporated' if weight_i != -1 else 'not found'))
        
        for line in f:
            if line[0] == '#':
                continue
            vec = np.array(line.split(), dtype=np.float64)

            d = data - vec[theory_i]
            new_like = -np.einsum('i,ij,j', d, prec, d)/2
            log_is_weight = new_like - vec[like_i]

            weight = np.e**log_is_weight

            if weight_i != -1:
                w = vec[weight_i]

                weight *= w
                norm_fact += w
                total_is -= log_is_weight * w
            else:
                norm_fact += 1
                total_is += log_is_weight

            output.write('%e\t%e\r\n' % (new_like, weight))

        output.write('# <log_weight> = %f\r\n' % (float(total_is)/float(norm_fact)))

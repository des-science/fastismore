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

import numpy as np
import pandas as pd
from astropy.io import fits
import twopoint # from cosmosis/cosmosis-standard-library/2pt/
import configparser
import argparse

parser = argparse.ArgumentParser(description = 'This code computes importance weights for a data vector given a chain with data_vector--2pt_theory_### columns')

parser.add_argument('chain', help = 'Base chain filename.')
parser.add_argument('data_vector', help = 'Data vector filename.')
parser.add_argument('output', help = 'Output importance sampling weights.')

parser.add_argument('--like_section', dest = 'like_section',
                       default = '2pt_like', required = False,
                       help = 'The 2pt_like section name used in cosmosis. Needed in order to get scale cuts.')

parser.add_argument('--datasets', dest = 'data_sets',
                       default = 'xip,xim,gammat,wtheta', required = False,
                       help = 'Data set names used in the data vector file.')

args = parser.parse_args()

data_sets = args.data_sets.split(',')

# ---------- Do scale cuts -------------

with open(args.chain) as f:
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

data_vector = twopoint.TwoPointFile.from_fits(args.data_vector, 'covmat')
for name in data_sets: 
    s = data_vector.get_spectrum(name)
    for b1, b2 in s.bin_pairs:
        option_name = "angle_range_{}_{}_{}".format(name, b1, b2)
        r = [float(e) for e in values.get(args.like_section, option_name).split()]
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

like_i   = np.where(labels == 'like')[0]   if 'like'   in labels else -1
prior_i  = np.where(labels == 'prior')[0]  if 'prior'  in labels else -1
post_i   = np.where(labels == 'post')[0]   if 'post'   in labels else -1
weight_i = np.where(labels == 'weight')[0] if 'weight' in labels else -1
theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])

assert like_i != -1 or (prior_i != -1 and post_i != -1)

total_is = 0.
norm_fact = 0.

print('Evaluating likelihoods...')

with open(args.chain) as f:

    with open(args.output, 'w+') as output:
        output.write('#new_like\tweight\r\n')
        output.write('#\r\n')
        output.write('# Importance sampling weights\r\n')
        output.write('# Chain: {}\r\n'.format(args.chain))
        output.write('# Data vector: {}\r\n'.format(args.data_vector))
        output.write('# Data vector size: {}\r\n'.format(np.sum(theory_i)))
        output.write('# Previous weights {}.\r\n'.format('were found and incorporated' if weight_i != -1 else 'not found'))
        
        for line in f:
            if line[0] == '#':
                continue
            vec = np.array(line.split(), dtype=np.float64)

            d = data - vec[theory_i]
            new_like = -np.einsum('i,ij,j', d, prec, d)/2
            log_is_weight = new_like - (vec[like_i] if like_i != -1 else vec[post_i]-vec[prior_i])

            weight = np.e**log_is_weight

            if weight_i != -1:
                w = vec[weight_i]

                weight *= w
                norm_fact += w
                total_is -= log_is_weight * w
            else:
                norm_fact += 1
                total_is -= log_is_weight

            output.write('%e\t%e\r\n' % (new_like, weight))

        output.write('# <log_weight> = %f\r\n' % (total_is/norm_fact))

#!/usr/bin/env python
# 
# author: Otavio Alves
#
# description: This code computes importance weights for a data vector given a
# chain with data_vector--2pt_theory_### columns
# 
# usage: importance_sampling.py [-h] [--like_section LIKE_SECTION]
#                               [--include_norm] [--datasets DATA_SETS]
#                               chain data_vector output
# 
# positional arguments:
#   chain                 Base chain filename.
#   data_vector           Data vector filename.
#   output                Output importance sampling weights.
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --like_section LIKE_SECTION
#                         The 2pt_like section name used in cosmosis. Needed in
#                         order to get scale cuts (default: 2pt_like).
#   --include_norm        Include normalization |C| in the likelihood
#                         (recommended if you are varying the covariance
#                         matrix).
#   --datasets DATA_SETS  Data set names used in the data vector file (default:
#                         xip,xim,gammat,wtheta).
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
                       help = 'The 2pt_like section name used in cosmosis. Needed in order to get scale cuts (default: 2pt_like).')

parser.add_argument('--include_norm', dest = 'include_norm', action='store_true',
                       help = 'Include normalization |C| in the likelihood (recommended if you are varying the covariance matrix).')

parser.add_argument('--datasets', dest = 'data_sets',
                       default = 'xip,xim,gammat,wtheta', required = False,
                       help = 'Data set names used in the data vector file (default: xip,xim,gammat,wtheta).')
# SJ begin
parser.add_argument('--like2pt', dest = 'name_2ptlike',
                       default = 'like', required = False,
                       help ='LIKELIHOODS--2PT_LIKE if chain was run with external data sets')
# SJ end


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

inifile = u'\n'.join(inifile)

print('Found PARAMS_INI')
values = configparser.ConfigParser(strict=False)
## begin NW. Make compatible with Python 2.7
import sys
PY3 = sys.version_info[0] == 3
if not PY3:
    import StringIO
    buf = StringIO.StringIO(inifile)
    values.readfp(buf)
else:
    ## end NW
    values.read_string(inifile) #Python3
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
    # ! SJ begin ! #
    nbinpairs = len(s.get_bin_pairs())
    if nbinpairs == 0 : pass
    else : 
	e.append(np.concatenate([s.get_pair(*p)[1] for p in s.get_bin_pairs()]))
    #e.append(np.concatenate([s.get_pair(*p)[1] for p in s.get_bin_pairs()]))
    # ! SJ end ! #

data = np.concatenate(e)

# -------------- Scale cuts done --------------

prec = np.linalg.inv(data_vector.covmat)

if args.include_norm:
    sign, log_det = np.linalg.slogdet(data_vector.covmat)

# SJ begin
#like_i   = np.where(labels == 'like')[0]   if 'like'   in labels else -1
like_i   = np.where(labels == args.name_2ptlike)[0]  if args.name_2ptlike in labels else -1
# SJ end
prior_i  = np.where(labels == 'prior')[0]  if 'prior'  in labels else -1
post_i   = np.where(labels == 'post')[0]   if 'post'   in labels else -1
weight_i = np.where(labels == 'weight')[0] if 'weight' in labels else -1
theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])
chi2_i   = np.where(labels == 'data_vector--2pt_chi2')[0] if 'data_vector--2pt_chi2' in labels else -1


## NW
if theory_i.sum() == 0:
    raise Exception("No theory vector columns found! Ensure your baseline chain has theory vector of form 'data_vector--2pt_theory_XX'?")
else:
    pass
## end

assert chi2_i != -1 or like_i != -1 or (prior_i != -1 and post_i != -1)

total_is = 0.
norm_fact = 0.


print('Evaluating likelihoods...')

with open(args.chain) as f:

    with open(args.output, 'w+') as output:
        output.write('#old_like\told_weight\tnew_like\tweight\r\n')
        output.write('#\r\n')
        output.write('# Importance sampling weights\r\n')
        output.write('# Chain: {}\r\n'.format(args.chain))
        output.write('# Data vector: {}\r\n'.format(args.data_vector))
        output.write('# Data vector size: {}\r\n'.format(np.sum(theory_i)))
        output.write('# Using {} column found in chain file.\r\n'.format('chi2' if chi2_i != -1 else ('like' if like_i != -1 else 'post')))
        if args.include_norm:
            output.write('# Including |C| factor in likelihood\r\n')
        if weight_i != -1:
            output.write('# Previous weights were found and incorporated\r\n')
        
        # Different ways to recover old_like depending on columns found
        if chi2_i != -1:
            print('Using -0.5*chi^2 column for log-like.')
            if args.include_norm:
                print('Including detC in normalization of loglike')
                old_like = lambda vec: -0.5*vec[chi2_i] - 0.5*log_det
            else:
                old_like = lambda vec: -0.5*vec[chi2_i]

        elif like_i != -1:
            print('Using like column.')
            old_like = lambda vec: vec[like_i]
        else:
            print('Using post-prior for likelihood.')
            old_like = lambda vec: vec[post_i] - vec[prior_i]
		
        loglikediff = []
        oldweights = []
		
        # Iterate through lines to compute IS weights (can probably rewrite this as array funcs but not necessary)
        for line in f:
            if line[0] == '#':
                continue
            vec = np.array(line.split(), dtype=np.float64)
            d = data - vec[theory_i]
            new_like = -np.einsum('i,ij,j', d, prec, d)/2

            if args.include_norm:
                new_like += -0.5*log_det

            log_is_weight = new_like - old_like(vec) 
            loglikediff.append(log_is_weight)
			
            weight = np.e**log_is_weight #change in prob
            if weight_i != -1:
                w_old = vec[weight_i] #old weight from baseline chain
				
                weight *= w_old #new weight
                norm_fact += w_old
                total_is -= log_is_weight * w_old
            else:
                w_old = 1.
                norm_fact += 1
                total_is -= log_is_weight
            oldweights.append(w_old)

            output.write('%e\t%e\t%e\t%e\r\n' % (old_like(vec), w_old, new_like, weight))

        output.write('# <log_weight> = %f\r\n' % (total_is/norm_fact))
        oldweights = np.array(oldweights)
        loglikediff = np.array(loglikediff)
        # print 'weighted average difference of logposterior: ', total_is/norm_fact
        print 'Weighted average difference of logposterior: ', -np.average(loglikediff, weights=oldweights)
        print 'Weighted RMS difference of logposterior: ', np.average(loglikediff**2, weights=oldweights)**0.5

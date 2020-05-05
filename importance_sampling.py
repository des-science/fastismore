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
import argparse, configparser
import sys, os

# Imports 2pt_like module
sys.path.append(os.environ['COSMOSIS_SRC_DIR'] + '/cosmosis-standard-library/likelihood/2pt')
twopointlike = __import__('2pt_like_allmarg')

class Params():
    """This just mimicks the SectionOption class used by cosmosis to read values from params.ini"""
    def __init__(self, filename, section):
        self.parser = self.load_ini(filename, 'params')
        self.section = section
        
    def load_ini(self, filename, ini=None):
        """Loads given ini info from chain file. If ini=None, loads directly from file.ini"""
        parser = configparser.ConfigParser(strict=False)

        if ini is None:
            parser.read_file(filename)
        else:
            ini = ini.upper()

            with open(filename) as f:
                line = f.readline()
                lines=[]

                print('Looking for {} file...'.format(ini))
                while("START_OF_{}".format(ini) not in line):
                    line = f.readline()

                print('{} file found!'.format(ini))
                while("END_OF_{}".format(ini) not in line):
                    line = f.readline()
                    lines.append(line.replace('#', ''))

                inifile = '\r'.join(lines[:-1])

                assert len(inifile) > 0

                parser.read_string(inifile)

        return parser

    def has_value(self, name):
        return self.parser.has_option(self.section, name)

    def get(self, name, default=None):
        if self.has_value(name):
            return self.parser.get(self.section, name)
        elif default is not None:
            return default
        else:
            raise Exception('Option {} at section {} not found.'.format(name, self.section))

    def get_int(self, *args, **kwargs):
        return int(self.get(*args, **kwargs))

    def get_bool(self, *args, **kwargs):
        return bool(self.get(*args, **kwargs))

    def get_double(self, *args, **kwargs):
        return np.double(self.get(*args, **kwargs))

    def get_string(self, *args, **kwargs):
        return str(self.get(*args, **kwargs))

    def get_double(self, *args, **kwargs):
        return np.double(self.get(*args, **kwargs))

    def get_double_array_1d(self, *args, **kwargs):
        return np.array(self.get(*args, **kwargs).split(), dtype=np.double)

    def __getattr__(self, attr):
        """defaults any unspecified methods to the ConfigParser object"""
        return getattr(self.parser, attr)

def load_labels(filename):
    """ Reads labels from first line in given file"""
    with open(filename) as f:
        labels = np.array(f.readline()[1:-1].lower().split())
    
    return labels

def main():
    # First, let's handle input arguments
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

    labels = load_labels(args.chain)

    # Loads params from chain file header
    params = Params(args.chain, args.like_section)

    # Sets data file to the specified one
    params.set(args.like_section, 'data_file', args.data_vector)

    # Loads the likelihood object building the data vector and covariance
    like_obj = twopointlike.TwoPointGammatMargLikelihood(params)

    # Gets data vector and inverse covariance from likelihood object
    data_vector = np.atleast_1d(like_obj.data_y)
    precision_matrix = like_obj.inv_cov

    if args.include_norm:
        sign, log_det = np.linalg.slogdet(data_vector.covmat)



    # SJ begin
    like_i   = np.where(labels == args.name_2ptlike)[0]  if args.name_2ptlike in labels else -1
    # SJ end
    weight_i = np.where(labels == 'weight')[0] if 'weight' in labels else -1
    theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])

    if like_i == -1:
        raise Exception("Likelihood column {} not found.".format(args.name_2ptlike))
    else:
        pass

    ## NW begin
    if theory_i.sum() == 0:
        raise Exception("No theory vector columns found! Ensure your baseline chain has theory vector of form 'data_vector--2pt_theory_XX'?")
    else:
        pass
    ## NW end

    total_is = 0.
    norm_fact = 0.

    print('Evaluating likelihoods...')

    with open(args.chain) as f:

        with open(args.output, 'w+') as output:
            # Setting the header of the output file
            output.write('#old_like\told_weight\tnew_like\tweight\r\n')
            output.write('#\r\n')
            output.write('# Importance sampling weights\r\n')
            output.write('# Chain: {}\r\n'.format(args.chain))
            output.write('# Data vector: {}\r\n'.format(args.data_vector))
            output.write('# Data vector size: {}\r\n'.format(np.sum(theory_i)))

            if args.include_norm:
                output.write('# Including detC factor in likelihood\r\n')

            if weight_i != -1:
                output.write('# Previous weights were found and incorporated in weight column\r\n')
            
            # Defines how we extract likelihood from the chain file
            old_like = lambda vec: vec[like_i]
            
            loglikediff = []
            oldweights = []
            
            # Iterate through lines to compute IS weights (can probably rewrite this as array funcs but not necessary)
            for line in f:
                if line[0] == '#':
                    continue
                vec = np.array(line.split(), dtype=np.float64)
                d = data_vector - vec[theory_i]
                new_like = -np.einsum('i,ij,j', d, precision_matrix, d)/2

                if args.include_norm:
                    new_like += -0.5*log_det

                log_is_weight = new_like - old_like(vec) 
                loglikediff.append(log_is_weight)
                
                ## Try to get ratio of likelihoods. If old likelihood is tiny, then we could be dividing two tiny numbers.
                ## If the old chain point has essentially 0 for the weight, then set new point to 0 as well.
                ## otherwise set to NaN 
                # try:
                #     likeratio = np.e**log_is_weight #change in prob
                # except OverflowError: 
                #     if (weight_i != -1) and vec[weight_i] < 1.e-300:
                #         likeratio = 0
                #     else:
                #         likeratio = NaN
                        
                #avoid dividing tiny numbers
                if (weight_i != -1) and vec[weight_i] < 1.e-300:
                    likeratio = 0
                else:
                    likeratio = np.e**log_is_weight #change in prob
                        
                if weight_i != -1:
                    w_old = vec[weight_i] #old weight from baseline chain
                    
                    weight = likeratio * w_old #new weight
                    norm_fact += w_old
                    total_is -= log_is_weight * w_old
                else:
                    w_old = 1.
                    weight = likeratio * w_old #new weight
                    norm_fact += 1
                    total_is -= log_is_weight
                oldweights.append(w_old)

                output.write('%e\t%e\t%e\t%e\r\n' % (old_like(vec), w_old, new_like, weight))

            output.write('# <log_weight> = %f\r\n' % (total_is/norm_fact))

            oldweights = np.array(oldweights)
            loglikediff = np.array(loglikediff)
            Nsample = len(loglikediff)
            
            loglikediff_mean = np.average(loglikediff, weights=oldweights)
            loglikediff_rms = np.average(loglikediff**2, weights=oldweights)**0.5
            
            #calc importance sampling effective sample size.
            weight_ratio = np.exp(loglikediff)
            normed_weights = weight_ratio / (Nsample * np.average(weight_ratio, weights=oldweights)) #really doing weighted sum
            eff_sample_frac = 1./(Nsample * np.average(normed_weights**2, weights=oldweights))

            
            #TODO: ESS using e^loglikediff and normalized correctly.
            print('Weighted average difference of logposterior: ', -loglikediff_mean) #this should match total_is/normfact
            print('Check: same result, using individual calculation: ', total_is/norm_fact)
            print('Weighted RMS difference of logposterior: ', loglikediff_rms)
            print('Baseline Sample Size     = {}'.format(Nsample))

            output.write('# RMS(log_weight) = %f\r\n' % (loglikediff_rms))
    #        output.write('# Effective sample size = %f\r\n' % (Nsample_eff_IS/Nsample))

if __name__ == '__main__':
    main()
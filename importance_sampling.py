#!/usr/bin/env python
# 
# author: Otavio Alves
#
# description: This code computes importance weights for a data vector given a
# chain with data_vector--2pt_theory_### columns
# 
# output: output.txt: -1/2*chi2 (log-likelihood) and weight for each point in chain.txt
#

import numpy as np
import argparse, configparser
import sys, os

#  Imports 2pt_like module
try:
    sys.path.append(os.environ['COSMOSIS_SRC_DIR'] + '/cosmosis-standard-library/likelihood/2pt')
except:
    print("Failed to find COSMOSIS dir. Did you set up COSMOSIS?")
    sys.exit(1)

twopointlike_allmarg = __import__('2pt_like_allmarg')
# twopointlike = __import__('2pt_like')

class Block():
    def __init__(self, labels, row):
        self.labels = labels
        self.row = row
        return

    def get_double(self, section, option):
        label = '--'.join([section, option])
#        print("Asking block for {}".format(label))
        index = np.where(self.labels == label)[0]
#        print("Value {}".format(self.row[index]))
        return self.row[index]

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

def main():
    # First, let's handle input arguments
    parser = argparse.ArgumentParser(description = 'This code computes importance weights for a data vector given a chain with data_vector--2pt_theory_### columns')

    parser.add_argument('chain', help = 'Base chain filename.')
    parser.add_argument('data_vector', help = 'Data vector filename.')
    parser.add_argument('output', help = 'Output importance sampling weights.')

    parser.add_argument('--like-section', dest = 'like_section',
                           default = '2pt_like', required = False,
                           help = 'The 2pt_like configuration section name used in the baseline chain. (default: 2pt_like).')

    # SJ begin
    parser.add_argument('--like-column', dest = 'like_column',
                           default = 'like', required = False,
                           help ='Likelihood column name in the baseline chain. (LIKELIHOODS--2PT_LIKE if chain was run with external data sets)')
    # SJ end

    parser.add_argument('--use-chi2', dest = 'chi2', action='store_true',
                           help = 'Treat like column as chi2, multiplying it by an additional -0.5 factor (useful when using chi2 column instead of likelihood).')

    parser.add_argument('--include-norm', dest = 'include_norm', action='store_true',
                           help = 'Force include_norm option.')

    args = parser.parse_args()

    # Load labels from chain file
    with open(args.chain) as f:
        labels = np.array(f.readline()[1:-1].lower().split())

    # Loads params from chain file header
    params = Params(args.chain, args.like_section)

    # Sets data file to the specified one
    params.set(args.like_section, 'data_file', args.data_vector)

    # Loads the likelihood object building the data vector and covariance
    like_obj = twopointlike_allmarg.TwoPointGammatMargLikelihood(params)
    # like_obj = twopointlike.TwoPointLikelihood(params)

    # Gets data vector and inverse covariance from likelihood object
    data_vector = np.atleast_1d(like_obj.data_y)

    include_norm = params.get_string('include_norm', default='F').lower() in ['true', 't', 'yes'] or args.include_norm
    include_norm = True


    # SJ begin
    like_i   = np.where(labels == args.like_column)[0]  if args.like_column in labels else -1
    # SJ end
    prior_i   = np.where(labels == 'prior')[0]  if 'prior' in labels else -1
    post_i   = np.where(labels == 'post')[0]  if 'post' in labels else -1
    weight_i = np.where(labels == 'weight')[0] if 'weight' in labels else -1
    theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])


    if like_i == -1:
        raise Exception("Likelihood column {} not found.".format(args.like_column))
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
            output.write('# Data vector size (from base chain): {}\r\n'.format(np.sum(theory_i)))

            if include_norm:
                output.write('# Including detC factor in likelihood\r\n')

            if weight_i != -1:
                output.write('# Previous weights were found and incorporated in weight column\r\n')
            
            # Defines how we extract likelihood from the chain file
            if args.chi2:
                old_like = lambda vec: -0.5*vec[like_i]
            elif args.like_column == 'post' and post_i != -1 and prior_i != -1:
                print("Using posterior")
                old_like = lambda vec: vec[post_i] - vec[prior_i]
            else:
                old_like = lambda vec: vec[like_i]
            
            loglikediff = []
            oldweights = []
            weights = []

            # Iterate through lines to compute IS weights (manually splitting lines to minimize use of RAM)
            for line in f:
                if line[0] == '#':
                    continue
                vec = np.array(line.split(), dtype=np.float64)

                block = Block(labels, vec)
                covariance_matrix = like_obj.extract_covariance(block)
                precision_matrix = like_obj.extract_inverse_covariance(block)

                d = data_vector - vec[theory_i]
                new_like = -np.einsum('i,ij,j', d, precision_matrix, d)/2

                if include_norm:
                    sign, log_det = np.linalg.slogdet(covariance_matrix)
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
                weights.append(weight)

                output.write('%e\t%e\t%e\t%e\r\n' % (old_like(vec), w_old, new_like, weight))

            oldweights = np.array(oldweights)
            weights = np.array(weights)
            loglikediff = np.array(loglikediff)
            Nsample = len(loglikediff)
            
            print(oldweights.shape, loglikediff.shape)
            loglikediff_mean = np.average(loglikediff, weights=oldweights, axis=0)
            loglikediff_rms = np.average(loglikediff**2, weights=oldweights, axis=0)**0.5
            
            #calc importance sampling effective sample size.
            weight_ratio = np.exp(loglikediff)
            normed_weights = weight_ratio / (Nsample * np.average(weight_ratio, weights=oldweights, axis=0)) #really doing weighted sum
            eff_sample_frac = 1./(Nsample * np.average(normed_weights**2, weights=oldweights, axis=0))

            base_ess = oldweights.sum()**2/(oldweights**2).sum()
            final_ess = weights.sum()**2/(weights**2).sum()

            print()
            print('Finished!')

            def write_output(line=''):
                line = str(line)
                print(line)
                output.write('# ' + line + '\r\n')
                return line

            #TODO: ESS using e^loglikediff and normalized correctly.
            write_output()
            write_output('dloglike_mean: {}'.format(-loglikediff_mean)) #this should match total_is/normfact
            if not np.isclose(-loglikediff_mean, total_is/norm_fact, 1e-4):
                write_output('WARNING: same quantity, but using independent calculation: {}'.format(total_is/norm_fact))
            write_output('dloglike_rms: {}'.format(loglikediff_rms))
            write_output()
            write_output('ESS_baseline (assuming uncorrelated samples) = {}'.format(base_ess))
            write_output('ESS_IS = {}'.format(final_ess))
            write_output('- ratio = {}'.format(final_ess/base_ess)) # Ratio (variance inflation factor) = base_ess/final_ess
            write_output()
            write_output('ESS_IS_alt = {}'.format(eff_sample_frac))

if __name__ == '__main__':
    main()

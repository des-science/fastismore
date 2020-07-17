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

LIKE_COLUMN_PRIORITY = iter(['like', 'post', '2pt_like--chi2'])

#  Imports 2pt_like module
try:
    sys.path.append(os.environ['COSMOSIS_SRC_DIR'] + '/cosmosis-standard-library/likelihood/2pt')
except:
    print("Failed to find COSMOSIS dir. Did you set up COSMOSIS?")
    sys.exit(1)

twopointlike = __import__('2pt_like_allmarg')

class ImportanceSamplingLikelihood(twopointlike.TwoPointGammatMargLikelihood):
    def __init__(self, options):
        super(ImportanceSamplingLikelihood, self).__init__(options)

class Block():
    """ This class mimicks cosmosis' data block. The likelihood object reads from it."""
    def __init__(self, labels, like_column='like'):
        self.labels = labels

        # if like_column not found, look for others in the LIKE_COLUMN_PRIORITY order
        while like_column not in labels:
            print("Couldn't find column {}.".format(like_column))
            like_column = next(LIKE_COLUMN_PRIORITY)
            print("Looking for {}.".format(like_column))

        if like_column == 'post':
            print("Using old loglike = post - prior.")
            if 'post' not in labels:
                raise Exception("Couldn't find column: post.")
            if 'prior' not in labels:
                raise Exception("Couldn't find column: prior.")
            prior_i = np.where(labels == 'prior')[0]
            post_i = np.where(labels == 'post')[0]
            self._like = lambda vec: float(vec[post_i] - vec[prior_i])
        elif 'chi2' in like_column:
            print("Using old loglike = -0.5*{}.".format(like_column))
            if like_column not in labels:
                raise Exception("Couldn't find column: {}.".format(like_column))
            chi_i = np.where(labels == like_column)[0]
            self._like = lambda vec: float(-0.5*vec[chi_i])
        else:
            print("Using old loglike = {}.".format(like_column))
            if like_column not in labels:
                raise Exception("Couldn't find column: {}.".format(like_column))
            like_i = np.where(labels == like_column)[0]
            self._like = lambda vec: float(vec[like_i])

        if 'weight' in labels:
            weight_i = np.where(labels == 'weight')[0]
            self._weight = lambda vec: float(vec[weight_i])
            self.weighted = True
        else:
            self._weight = lambda vec: 1.0
            self.weighted = False

        theory_i = np.array(['data_vector--2pt_theory_' in l for l in labels])

        ## NW begin
        if theory_i.sum() == 0:
            raise Exception("No theory vector columns found! Ensure your baseline chain has theory vector of form 'data_vector--2pt_theory_XXX'?")
        else:
            self.theory_len = theory_i.sum()
            self._theory = lambda vec: vec[theory_i]
        ## NW end

        return

    def get_double(self, section, option):
        label = '--'.join([section, option])
        index = np.where(self.labels == label)[0]
        return self.row[index]

    def update(self, row):
        self.row = row
        return

    def get_theory(self):
        return self._theory(self.row)

    def get_like(self):
        return self._like(self.row)

    def get_weight(self):
        return self._weight(self.row)

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

def get_ess_dict(weights):
    w = weights/weights.sum()
    N = len(w)

    return {
            'Euclidean distance': 1/(w**2).sum(), # overestimates
            'Inverse maximum weight': 1/np.max(w), # underestimates; best when bias is large

            'Gini coefficient': -2*np.sum(np.arange(1,N+1)*np.sort(w)) + 2*N + 1, # best when bias and N are small
            'Square root sum': np.sum(np.sqrt(w))**2,
            'Peak integrated': -N*np.sum(w[w>=1/N]) + np.sum(w>=1/N) + N,
            'Shannon entropy': 2**(-np.sum(w[w>0]*np.log2(w[w>0]))),

            # Not stable
            # 'Maximum': N + 1 - N*np.max(w),
            # 'Peak count': np.sum(w>=1/N),
            # 'Minimum': N*(N - 1)*np.min(w) + 1,
            # 'Inverse minimum': 1/((1-N)*np.min(w) + 1),
            # 'Entropy': N - 1/np.log2(N)*(-np.sum(w[w>0]*np.log2(w[w>0]))),
            # 'Inverse entropy': -N*np.log2(N)/(-N*np.log2(N) + (N - 1)*(-np.sum(w[w>0]*np.log2(w[w>0])))),
            }

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
               help ='Likelihood column name in the baseline chain. (likelihoods--2pt_like if chain was run with external data sets)')
    # SJ end

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
    like_obj = ImportanceSamplingLikelihood(params)

    # Gets data vector and inverse covariance from likelihood object
    data_vector = np.atleast_1d(like_obj.data_y)

    # include_norm is true if covariance is not fixed
    include_norm = args.include_norm
    include_norm = include_norm or not like_obj.constant_covariance
    include_norm = include_norm or params.get_string('include_norm', default='F').lower() in ['true', 't', 'yes']

    block = Block(labels, args.like_column)

    # Initialize these variables
    covariance_matrix = None
    precision_matrix = None
    log_det = None

    total_is = 0.
    norm_fact = 0.

    print('Evaluating likelihoods...')

    with open(args.chain) as f:

        with open(args.output, 'w+') as output:
            # Setting the header of the output file
            output.write('#old_like\told_weight\tnew_like\tweight\n')
            output.write('#\n')
            output.write('# Importance sampling weights\n')
            output.write('# Chain: {}\n'.format(args.chain))
            output.write('# Data vector: {}\n'.format(args.data_vector))
            output.write('# Data vector size (from base chain): {}\n'.format(block.theory_len))

            if include_norm:
                output.write('# Including detC factor in likelihood\n')

            if block.weighted:
                output.write('# Previous weights were found and incorporated in weight column\n')

            # We'll keep track of these quantities
            log_is_weights = []
            old_weights = []
            weights = []

            # Iterate through lines to compute IS weights (manually splitting lines to minimize use of RAM)
            for line in f:
                if line[0] == '#':
                    continue

                block.update(np.array(line.split(), dtype=np.float64))

                # Check if covariance is set and whether we need to constantly update it
                if not like_obj.constant_covariance or covariance_matrix is None:
                    covariance_matrix = like_obj.extract_covariance(block)
                    precision_matrix = like_obj.extract_inverse_covariance(block)

                # Core computation
                d = data_vector - block.get_theory()
                new_like = -np.einsum('i,ij,j', d, precision_matrix, d)/2

                if include_norm :
                    # Check if log_det is set and whether we need to constantly update it
                    if not like_obj.constant_covariance or log_det is None:
                        log_det = like_obj.extract_covariance_log_determinant(block)
                    new_like += -0.5*log_det

                log_is_weight = new_like - block.get_like()
                weight = np.nan_to_num(np.exp(log_is_weight))

                if block.weighted:
                    weight = np.nan_to_num(weight*block.get_weight())
                    old_weights.append(block.get_weight())

                log_is_weights.append(log_is_weight)
                weights.append(weight)

                output.write('%e\t%e\t%e\t%e\n' % (block.get_like(), block.get_weight(), new_like, weight))

            print()
            print('Finished!')

            # Now let's compute diagnostic stats
            weights = np.array(weights)
            old_weights = np.array(old_weights) if len(old_weights) > 0 else np.ones_like(weights)
            log_is_weights = np.array(log_is_weights)
            Nsample = len(log_is_weights)

            log_is_weights_mean = np.average(-log_is_weights, weights=old_weights)
            log_is_weights_rms = np.average(log_is_weights**2, weights=old_weights)**0.5

            def write_output(line=''):
                line = str(line)
                print(line)
                output.write('# ' + line + '\n')
                return line

            write_output('Delta loglike')
            write_output('\tAverage: {:7n}'.format(log_is_weights_mean))
            write_output('\tRMS:     {:7n}'.format(log_is_weights_rms))
            write_output()
            write_output('Effective sample sizes')

            ESS_base = get_ess_dict(old_weights)
            ESS_IS = get_ess_dict(weights)

            for key in ESS_base.keys():
                write_output('\t{:<30}\t{:7n}/{:7n} = {:7n}'.format(key, ESS_IS[key], ESS_base[key], ESS_IS[key]/ESS_base[key]))

            write_output()
            write_output('\tTotal samples' + ' '*27 + '{}'.format(Nsample))


if __name__ == '__main__':
    main()

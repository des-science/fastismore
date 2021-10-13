#!/usr/bin/env python
#
# author: Otavio Alves, Noah Weaverdyck
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
    sys.path.append(os.environ['COSMOSIS_SRC_DIR'])
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
            
        self.like_column = like_column
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
        value = self.get(*args, **kwargs)
        if type(value) == bool:
            return value
        if type(value) == str:
            value = value.lower()
            if value in ['y', 'yes', 't','true']:
                return True
            elif value in ['n', 'no', 'f', 'false']:
                return False

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

def pc_to_cosmosis_sample(pc_sample_list, cosmosis_labels):
    """ Pass row from polychord chain and return in cosmosis format.
    pc file cols are: weight, minus2like, [all params], prior
    Cosmosis cols are (usually): [all params], prior, like, post, weight.
    """
    #make sure cosmosis has all the column labels we are assuming
    for label in ['prior', 'like', 'post', 'weight']:
        assert label in cosmosis_labels, label
#     print(pc_sample_list)
    pc_sample_list = np.float64(pc_sample_list)
    like = -0.5*pc_sample_list[1]
    prior = pc_sample_list[-1]
    post = prior + like
    weight = pc_sample_list[0]
    
    cosmosis_sample_list = list(pc_sample_list[2:]) + [like, post, weight]
    return cosmosis_sample_list
            
def load_boosted_data(cosmosis_chain_fn):
    """load and store data from the polychord output chain (which includes more samples if it was run with boost_posteriors=T) instead of the cosmosis chain.
    Retrieved using `self.filename_boosted`, which by default is at './pcfiles/pc_[cosmosis chain filename]_.txt' relative to the cosmosis chain referenced in self.filename. """
    data = []
    with open(cosmosis_chain_fn) as f: ##get labels and mask from cosmosis chain output
        labels_cosmosis = np.array(f.readline()[1:-1].lower().split())
        mask_cosmosis = ["data_vector" not in l for l in labels_cosmosis] #for size reasons, don't load data_vector terms
        mask_cosmosis_params = [("data_vector" not in l) and (l not in ['weight', 'like', 'prior', 'post']) for l in labels_cosmosis]
        mask_cosmosis_params_dv = [l not in ['weight', 'like', 'prior', 'post'] for l in labels_cosmosis]

        labels_pc = np.array(['weight', 'like'] + list(labels_cosmosis[mask_cosmosis_params_dv]) + ['prior']) #order of columns in PC output files
        mask_pc = ["data_vector" not in l for l in labels_pc]
    # boosted_data = np.genfromtxt(self.filename_boosted)[:,:] #array too large

    with open(self.filename_boosted) as f:
        for line in f.readlines():
            if '#' in line:
                continue
            else:
                data.append(np.array(line.split(), dtype=np.double)[mask_pc])

    self.data = {labels_pc[mask_pc][i].lower(): col for i, col in enumerate(np.array(data).T)}
    self.data['like'] = -0.5 * self.data['like'] # PC originally stores -2*loglike, change to just loglike to match cosmosis
    self.data['post'] = self.data['prior'] + self.data['like']
    self.N = len(self.data[labels_pc[0]])
    return self.data

def importance_sample(bl_chain_fn, data_vector_file, output_fn, like_section='2pt_like', like_column='like', include_norm=False, pc_chain_fn=None, max_samples=1e9):
    """This code computes importance weights for a data vector given a chain with data_vector--2pt_theory_### columns. It saves an output file with weights and likelihoods for samples of both the baseline (old) and importance sampled (new) chains.
    
    Parameters:
    bl_chain_fn (str): Base chain filename
    data_vector_file (str): Data vector filename
    output_fn (str): Filename of output with new likelihoods and weights
    like_section (str): The 2pt_like configuration section name used in the baseline chain. (default: 2pt_like).
    like_column (str): Likelihood column name in the baseline chain. (likelihoods--2pt_like if chain was run with external data sets)
    include_norm (bool): Force inclusion of the covariance norm in likelihood evaluation (default=False)
    pc_chain_fn (str): Optional filepath to the polychord chain output. If included, load the baseline chain from the polychord output files rather than cosmosis output (useful if boost_posterior=T).
    max_samples (int): Max number of samples to run (useful for debugging). Default 1e9.

    Returns:
    dict: Containing keys 'old_weights', 'new_weights', 'old_likes', 'new_likes'
    """

    # Load labels from chain file
    with open(bl_chain_fn) as f:
        labels = np.array(f.readline()[1:-1].lower().split())

    # Loads params from chain file header
    params = Params(bl_chain_fn, like_section)

    # Sets data file to the specified one
    params.set(like_section, 'data_file', data_vector_file)

    # Loads the likelihood object building the data vector and covariance
    like_obj = ImportanceSamplingLikelihood(params)

    # Gets data vector and inverse covariance from likelihood object
    data_vector = np.atleast_1d(like_obj.data_y)

    # include_norm is true if covariance is not fixed
#     include_norm = args.include_norm
    include_norm = include_norm or not like_obj.constant_covariance
    include_norm = include_norm or params.get_bool('include_norm', default=False)

    block = Block(labels, like_column)

    # Initialize these variables
    precision_matrix = None
#     log_det = None
    _, log_det_orig = np.linalg.slogdet(like_obj.cov_orig)

    total_is = 0.
    norm_fact = 0.

    print('Evaluating likelihoods...')

    #
    if pc_chain_fn:
        samples_fn = pc_chain_fn
        with open(bl_chain_fn) as f: #get lables from cosmosis chain.
            cosmosis_labels = np.array(f.readline()[1:-1].lower().split())
    else:
        samples_fn = bl_chain_fn
        
    with open(samples_fn) as f:

        with open(output_fn, 'w+') as output:
            # Setting the header of the output file
            output.write('#old_like\told_weight\tnew_like\tweight\n')
            output.write('#\n')
            output.write('# Importance sampling weights\n')
            output.write('# Chain: {}\n'.format(samples_fn))
            output.write('# Data vector: {}\n'.format(data_vector_file))
            output.write('# Data vector size (from base chain): {}\n'.format(block.theory_len))
            output.write('# Like_column = {}\n'.format(block.like_column))
            if include_norm:
                output.write('# Including detC factor in likelihood\n')

            if block.weighted:
                output.write('# Previous weights were found and incorporated in weight column\n')

            # We'll keep track of these quantities
            log_is_weights = []
            old_weights = []
            weights = []
            old_likes = []
            new_likes = []

            # Iterate through lines to compute IS weights (manually splitting lines to minimize use of RAM)
            for ii,line in enumerate(f):

                if line[0] == '#':
                    continue
                mysample = line.split() if not pc_chain_fn else pc_to_cosmosis_sample(line.split(), cosmosis_labels)
                block.update(np.array(mysample, dtype=np.float64))

                # Check if covariance is set and whether we need to constantly update it
                if not like_obj.constant_covariance or precision_matrix is None:
#                     covariance_matrix = like_obj.extract_covariance(block) #slow. recomputes cholesky of cov_orig each time
#                     covariance_matrix = (like_obj.cov_orig)[:]
                    precision_matrix = like_obj.extract_inverse_covariance(block)

                # Core computation
                d = data_vector - block.get_theory()
                new_like = -np.einsum('i,ij,j', d, precision_matrix, d)/2

                if include_norm :
                    # Check if log_det is set and whether we need to constantly update it
                    if not like_obj.constant_covariance:
                        log_det = log_det_orig + like_obj.logdet_fac
#                         log_det = like_obj.extract_covariance_log_determinant(block)
                    else:
                        log_det = log_det_orig
                        
                    new_like += -0.5*log_det

                old_like = block.get_like()
                old_weight = block.get_weight()
                log_is_weight = new_like - old_like
                weight = np.nan_to_num(np.exp(log_is_weight))

                if block.weighted:
                    weight = np.nan_to_num(weight*old_weight) #new weighting for sample
                    old_weights.append(old_weight)

                log_is_weights.append(log_is_weight)
                weights.append(weight)
                old_likes.append(old_like)
                new_likes.append(new_like)

                output.write('%e\t%e\t%e\t%e\n' % (old_like, old_weight, new_like, weight))
                if ii%10000==0:
                    print('{} evals done...'.format(ii))
                if ii>max_samples:
                    print('Reached max samples passed by user ({})'.format(max_samples))
                    output.write('Halted because reached max samples passed by user: {}'.format(max_samples))
                    break

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
            return {'old_weights':old_weights, 'new_weights':weights, 'old_like':np.array(old_likes), 'new_likes':np.array(new_likes)}

if __name__ == '__main__':
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
    #NW
    parser.add_argument('--pc-chain-fn', dest = 'pc_chain_fn', required = False,
                    help = 'Optional filepath to the polychord chain output. If included, load the baseline chain from the polychord output files rather than cosmosis output (useful if boost_posterior=T).')

    parser.add_argument('--max-samples', dest = 'max_samples', type=int, default = 999999999, required = False,
               help = 'Max number of samples before exiting (for debugging purposes).')
    
    args = parser.parse_args()
    
    
    importance_sample(bl_chain_fn=args.chain, 
                      data_vector_file=args.data_vector, 
                      like_section=args.like_section,
                      like_column=args.like_column,
                      output_fn = args.output,
                      pc_chain_fn=args.pc_chain_fn,
                      max_samples=args.max_samples)

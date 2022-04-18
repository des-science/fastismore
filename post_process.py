#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
import matplotlib.pyplot as plot
from getdist import MCSamples, plots
import argparse, configparser, copy
import itertools as itt
import shivam_2d_bias
from chainconsumer import ChainConsumer

params2plot = [
   #  'cosmological_parameters--omega_m',
   # 'cosmological_parameters--s8',
   # 'cosmological_parameters--w',
   # 'cosmological_parameters--wa',
   # 'cosmological_parameters--wp',
    # 'cosmological_parameters--sigma_8',
#    'cosmological_parameters--h0',
#    'cosmological_parameters--omega_b',
#    'cosmological_parameters--n_s',
#    'cosmological_parameters--a_s',
   'cosmological_parameters--neff',
   'cosmological_parameters--meffsterile',
#    'shear_calibration_parameters--m1',
#    'shear_calibration_parameters--m2',
#    'shear_calibration_parameters--m3',
#    'shear_calibration_parameters--m4',
#    'lens_photoz_errors--bias_1',
#    'lens_photoz_errors--bias_2',
#    'lens_photoz_errors--bias_3',
#    'lens_photoz_errors--bias_4',
#    'lens_photoz_errors--bias_5',
#    'bias_lens--b1',
#    'bias_lens--b2',
#    'bias_lens--b3',
#    'bias_lens--b4',
#    'bias_lens--b5',
#    'intrinsic_alignment_parameters--a1',
#    'intrinsic_alignment_parameters--a2',
#    'intrinsic_alignment_parameters--alpha1',
#    'intrinsic_alignment_parameters--alpha2',
#    'intrinsic_alignment_parameters--bias_ta'
    # 'cosmological_parameters--omega_k',
    # 'npg_parameters--a2',
    # 'npg_parameters--a3',
    # 'npg_parameters--a4',
]

not_param = [
    'like',
    'old_like',
    'delta_loglike',
    'new_like',
    'new_prior',
    'new_post',
    'prior',
    'post',
    '2pt_like',
    'old_weight',
    'weight',
    'old_post',
    'log_weight',
]

label_dict = {
    'cosmological_parameters--tau':  r'\tau',
    'cosmological_parameters--w':  r'w',
    'cosmological_parameters--wa':  r'w_a',
    'cosmological_parameters--wp':  r'w_p',
    'cosmological_parameters--w0_fld':  r'w_{GDM}',
    'cosmological_parameters--cs2_fld': r'c_s^2',
    'cosmological_parameters--log_cs2': r'log(c_s^2)',
    'cosmological_parameters--omega_m': r'\Omega_m',
    'cosmological_parameters--omega_c': r'\Omega_c',
    'cosmological_parameters--ommh2': r'\Omega_m h^2',
    'cosmological_parameters--ombh2': r'\Omega_b h^2',
    'cosmological_parameters--omch2': r'\Omega_c h^2',
    'cosmological_parameters--h0':    r'h',
    'cosmological_parameters--omega_b': r'\Omega_b',
    'cosmological_parameters--n_s':  r'n_s',
    'cosmological_parameters--a_s':  r'A_s',
    'cosmological_parameters--omnuh2':  r'\Omega_{\nu}',
    'cosmological_parameters--sigma_8': r'\sigma_8',
    'cosmological_parameters--sigma_12': r'\sigma_12',
    'cosmological_parameters--s8': r'S_8',
    'cosmological_parameters--massive_nu': r'\nu_\text{massive}',
    'cosmological_parameters--massless_nu': r'\nu_\text{massless}',
    'cosmological_parameters--omega_k': r'\Omega_k',
    'cosmological_parameters--yhe': r'Y_\text{He}',

    'intrinsic_alignment_parameters--a': r'A_{IA}',
    'intrinsic_alignment_parameters--alpha': r'\alpha_{IA}',
    'bin_bias--b1': r'b_1',
    'bin_bias--b2': r'b_2',
    'bin_bias--b3': r'b_3',
    'bin_bias--b4': r'b_4',
    'bin_bias--b5': r'b_5',
    'shear_calibration_parameters--m1': r'm_1',
    'shear_calibration_parameters--m2': r'm_2',
    'shear_calibration_parameters--m3': r'm_3',
    'shear_calibration_parameters--m4': r'm_4',
    'lens_photoz_errors--bias_1': r'z^l_1',
    'lens_photoz_errors--bias_2': r'z^l_2',
    'lens_photoz_errors--bias_3': r'z^l_3',
    'lens_photoz_errors--bias_4': r'z^l_4',
    'lens_photoz_errors--bias_5': r'z^l_5',
    'wl_photoz_errors--bias_1': r'z^s_1',
    'wl_photoz_errors--bias_2': r'z^s_2',
    'wl_photoz_errors--bias_3': r'z^s_3',
    'wl_photoz_errors--bias_4': r'z^s_4',

    'bias_lens--b1': r'b_1',
    'bias_lens--b2': r'b_2',
    'bias_lens--b3': r'b_3',
    'bias_lens--b4': r'b_4',
    'bias_lens--b5': r'b_5',
    'intrinsic_alignment_parameters--a1': r'A_1',
    'intrinsic_alignment_parameters--a2': r'A_2',
    'intrinsic_alignment_parameters--alpha1': r'\alpha_1',
    'intrinsic_alignment_parameters--alpha2': r'\alpha_2',
    'intrinsic_alignment_parameters--bias_ta': r'b_ta',
    'intrinsic_alignment_parameters--z_piv': r'z_\text{piv}',

    'mag_alpha_lens--alpha_1': r'\alpha_\text{lens}^1',
    'mag_alpha_lens--alpha_2': r'\alpha_\text{lens}^2',
    'mag_alpha_lens--alpha_3': r'\alpha_\text{lens}^3',
    'mag_alpha_lens--alpha_4': r'\alpha_\text{lens}^4',
    'mag_alpha_lens--alpha_5': r'\alpha_\text{lens}^5',
    
    'npg_parameters--a1': 'A_1',
    'npg_parameters--a2': 'A_2',
    'npg_parameters--a3': 'A_3',
    'npg_parameters--a4': 'A_4',
}

param_to_label = np.vectorize(lambda param: label_dict[param] if param in label_dict else param)
param_to_latex = np.vectorize(lambda param: '${}$'.format(label_dict[param]) if param in label_dict else param)

def load_ini(filename, ini=None):
    """loads given ini info from chain file. If ini=None, loads directly from file.ini"""
    values = configparser.ConfigParser(strict=False)

    if ini is None:
        values.read_file(filename)
    else:
        ini = ini.upper()
        with open(filename) as f:
            line = f.readline()
            lines = []

            print("Looking for START_OF_{} in file {}".format(ini, filename))

            while("START_OF_{}".format(ini) not in line):
                line = f.readline()
                if line == '':
                    raise Exception('START_OF_{} not found in file {}.'.format(ini, filename))

            while("END_OF_{}".format(ini) not in line):
                line = f.readline()
                lines.append(line.replace('#', ''))
                if line == '':
                    raise Exception('END_OF_{} not found in file {}.'.format(ini, filename))


    values.read_string('\r'.join(lines[:-1]))

    return values

class Chain:
    """Description: Generic chain object"""

    def __init__(self, filename, boosted=False, weight_option=0):
        """Initialize chain with given filename (full path). Set boosted=True if you want to load a boosted chain. If boosted_chain_fn is passed, use that, otherwise use default format/path for Y3 (i.e. a subdir /pcfiles/ with 'pc_' added and 'chain' dropped.
        weight_option: 0: use column "weight" as weight for chain. [Default and almost certainly what you want. Use subclass ImportanceChain for importance weights.]
                       1: use exp('log_weight')*'old_weight' as weight for chain
                       2: use 'old_weight' as weight for chain
                       
            """
        self.filename = filename
        self.weight_option = int(weight_option)
        self.name = '.'.join(filename.split('/')[-1].split('.')[:-1])
        self.chaindir = '/'.join(filename.split('/')[:-1])
        self.filename_boosted = self.chaindir + '/pcfiles/pc' + self.name[5:] + '_.txt' #go to pcfiles subdir and drop 'chain' from beginning of name
        if boosted:
            self.load_boosted_data()
        else:
            self.load_data()
        self.add_extra()

    def load_data(self, boosted=False, nsample=0):
        data = []
        
        with open(self.filename) as f:
            labels = np.array(f.readline()[1:-1].lower().split())
            mask = ["data_vector" not in l for l in labels]
            for line in f.readlines():
                if '#nsample' in line:
                    nsample = int(line.replace('#nsample=', ''))
                    print(f'Found nsample = {nsample}')
                elif '#' in line:
                    continue
                else:
                    data.append(np.array(line.split(), dtype=np.double)[mask])
        if nsample != 0:
            self.nsample = nsample
        self.data = {labels[mask][i].lower(): col for i, col in enumerate(np.array(data)[-nsample:,:].T)}
        self.N = len(self.data[labels[0]])
        return self.data

    def load_boosted_data(self):
        """load and store data from the polychord output chain (which includes more samples if it was run with boost_posteriors=T) instead of the cosmosis chain.
        Retrieved using `self.filename_boosted`, which by default is at './pcfiles/pc_[cosmosis chain filename]_.txt' relative to the cosmosis chain referenced in self.filename. """
        data = []
        with open(self.filename) as f: ##get labels and mask from cosmosis chain output
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
 
    def add_extra(self, data=None, extra=None):
        if data == None:
            data = self.data
        else:
            data = copy.copy(data)

        if extra != None:
            data.update(extra)

        data['cosmological_parameters--s8'] = \
            data['cosmological_parameters--sigma_8']*(data['cosmological_parameters--omega_m']/0.3)**0.5

        data['cosmological_parameters--ommh2'] = \
            data['cosmological_parameters--omega_m']*data['cosmological_parameters--h0']**2

        data['cosmological_parameters--ombh2'] = \
            data['cosmological_parameters--omega_b']*data['cosmological_parameters--h0']**2

        data['cosmological_parameters--omch2'] = \
            data['cosmological_parameters--ommh2'] - data['cosmological_parameters--ombh2']

        data['cosmological_parameters--omega_c'] = \
            data['cosmological_parameters--omega_m'] - data['cosmological_parameters--omega_b']
        
        if 'cosmological_parameters--w' in data.keys() and 'cosmological_parameters--wa' in data.keys():
            w0, wa = data['cosmological_parameters--w'], data['cosmological_parameters--wa']
            w0wa_cov = np.cov(w0, wa, aweights=self.get_weights())
            ap = 1. + w0wa_cov[0,1]/w0wa_cov[1,1]
            print(f"a_pivot = {ap}")
            data['cosmological_parameters--wp'] = w0 + wa*(1. - ap)

        return data

    def get_sampler(self):
        """reads the sampler name from a given chain"""
        sampler = self.params().get('runtime', 'sampler')
        # print("Sampler is {}".format(sampler))
        return sampler

    def get_params(self):
        params = [l for l in self.data.keys() if l not in not_param]

        if len(params) > 0:
            return params
        else:
            raise Exception("No parameters found..")

    def get_labels(self, params=None):
        if params == None:
            params = self.get_params()
        return param_to_label(params)

    def on_params(self, params=None):
        if params == None:
            params = self.get_params()
        return np.array([self.data[l] for l in params]).T

    def get_fiducial(self, filename=None, extra=None):
        """loads range values from values.ini file or chain file"""

        fiducial = {p:
            float((lambda x: x[1] if len(x) == 3 else x[0])(self.values().get(*p.split('--')).split())) \
            if self.values().has_option(*p.split('--')) \
            else None \
            for p in self.get_params()}

        return self.add_extra(fiducial, extra)

    def params(self):
        if not hasattr(self, '_params'):
            self._params = load_ini(self.filename, ini='params')
        return self._params

    def values(self):
        if not hasattr(self, '_values'):
            self._values = load_ini(self.filename, ini='values')
        return self._values

    def get_ranges(self, filename=None, params=None):
        """loads range values from values.ini file or chain file"""
        
        if params == None:
            params = self.get_params()

        self.ranges = {p:
            (lambda x: [float(x[0]), float(x[2])] if len(x) == 3 else [None, None])(self.values().get(*p.split('--')).split()) \
            if self.values().has_option(*p.split('--')) \
            else [None, None] \
            for p in params}

        return self.ranges

    def get_MCSamples(self, settings=None, params=None):
        
        if params == None:
            params = self.get_params()

        if not hasattr(self, '_mcsamples'):
            self._mcsamples = MCSamples(
                samples=self.on_params(params=params),
                weights=self.get_weights(),
                #loglikes=self.get_likes(),

                ranges=self.get_ranges(params=params),
                sampler='nested' if self.get_sampler() in ['multinest', 'polychord'] else 'mcmc',

                names=params,
                labels=[l for l in self.get_labels(params=params)],
                settings=settings,
            )

        return self._mcsamples

    def get_weights(self):
        if self.weight_option == 0 and 'weight' in self.data.keys():
            print('WARNING: Using column "weight" as weight for baseline chain.')
            w = self.data['weight']
            return w/w.sum()
        elif self.weight_option == 1 and 'log_weight' in self.data.keys() and 'old_weight' in self.data.keys():
            print('WARNING: Using "exp(log_weight)*old_weight" as weight for baseline chain.')
            w = self.data['old_weight']
            return w/w.sum()
        elif self.weight_option == 2 and 'old_weight' in self.data.keys():
            print('WARNING: Using column "old_weight" as weight for baseline chain.')
            w = self.data['old_weight']
            return w/w.sum()
        else:
            print('No weight criteria satisfied. Not returning weights.')
            return None

    def get_likes(self):
        return self.data['like']

    def get_mean_err(self, params):
        return self.get_MCSamples().std(params)/self.get_ESS()**0.5

    def get_std(self, params):
        return self.get_MCSamples().std(params)

    def get_mean(self, params):
        return self.get_MCSamples().mean(params)

    def get_ESS(self):
        """compute and return effective sample size."""

        w = self.get_weights()
        return 1./(w**2).sum()

    def get_ESS_dict(self):
        """compute and return effective sample size."""
        if not hasattr(self, 'ESS_dict'):

            w = self.get_weights()
            N = len(w)

            self.ESS_dict = {
                    'Euclidean distance': 1/(w**2).sum(), # overestimates
                    'Inverse max weight': 1/np.max(w), # underestimates; best when bias is large
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
        return self.ESS_dict

class ImportanceChain(Chain):
    """Description: object to load the importance weights, plot and compute statistics.
       Should be initialized with reference to the respective baseline chain: ImportanceChain(base_chain)"""

    def __init__(self, filename, base_chain):
        self.filename = filename
        self.name = '.'.join(filename.split('/')[-1].split('.')[:-1])
        self.base = base_chain
        self.load_data()

    def get_dloglike_stats(self):
        """compute weighted average and rms of loglikelihood difference from baseline to IS chain. 
        Deviance is -2*loglike; for Gaussian likelihood, dloglike = -0.5 * <delta chi^2>.
        RMS is included for back-compatibility. It can capture some differences that dloglike misses, but these are largely captured by ESS, so
        dloglike and ESS should work as primary quality statistics.
        Should be <~ o(1).
        returns (dloglike, rms_dloglike)"""
        dloglike = np.average(self.data['new_like'] - self.data['old_like'], weights=self.data['old_weight'])
        rmsdloglike = np.average((self.data['new_like'] - self.data['old_like'])**2, weights=self.data['old_weight'])**0.5

        return dloglike, rmsdloglike

    def get_ESS_NW(self, weight_by_multiplicity=True):
        """compute and return effective sample size of is chain. 
        Is a little more correct than using euclidean ESS of final weights in how it treats the fact that initial chain is weighted, 
        so we shouldn't be able to get a higher effective sample size by adding additional weights. Difference is small in practice.
        If is chain is identical to baseline, then just equals full sample size.
        insensitive to multiplicative scaling, i.e. if IS chain shows all points exactly half as likely, will not show up in ess,
        use mean_dloglike stat for that.
        (see e.g. https://arxiv.org/pdf/1602.03572.pdf or 
        http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html)"""
        #Noisier if compute from new_weight/old_weight, so use e^dloglike directly.
        weight_ratio = np.exp(self.data['new_like'] - self.data['old_like'])
        nsamples = len(weight_ratio)
        if weight_by_multiplicity: 
            mult = self.data['old_weight']
        else:
            mult = np.ones_like(weight_ratio)
        normed_weights = weight_ratio / (np.average(weight_ratio, weights=mult)) #pulled out factor of nsamples
        return nsamples * 1./(np.average(normed_weights**2, weights=mult))

    def get_delta_logz(self):
        """get estimate on shift of evidence. Note that only uses posterior points; won't account for contributions from changes in  volume of likelihood shells"""
        w_is = self.get_is_weights()
        w_bl = self.base.get_weights()
        return np.log(np.average(w_is[w_bl>0], weights=w_bl[w_bl>0]))
    
    def get_mean_err(self, params):
        return self.base.get_MCSamples().std(params)/self.get_ESS()**0.5

    def get_mean(self, params):
        return self.get_MCSamples().mean(params)

    def on_params(self, *args, **kwargs):
        return self.base.on_params(*args, **kwargs)

    def on_params(self, params=None):
        if params == None:
            params = self.get_params()
        #data = self.base.data
        data = self.add_extra(self.base.data)
        data.update(self.data)
        
        return np.array([data[l] for l in params]).T
    
    def get_likes(self):
        return self.data['new_like']

    def get_is_weights(self, regularize=True):
        """If regularize=True (default), divide by maximum likelihood difference before computing weights. Cancels when normalize weights, and helps with overflow when large offset in loglikelihoods."""
        nonzero = (self.base.get_weights()!=0)
        likediff = self.data['new_like'] - self.data['old_like']
        if regularize==True:
            maxdiff = np.max(likediff[nonzero])
        w = np.nan_to_num(np.exp(likediff - maxdiff))
        if 'extra_is_weight' in self.data.keys():
            print('Using extra IS weight.')
            w *= self.data['extra_is_weight']
        return w

    def get_weights(self):
        # w = np.nan_to_num(self.data['old_weight']*self.get_is_weights())
        print("WARNING: getting IS weights.")
        w_bl = self.base.get_weights()
        w = np.zeros_like(w_bl)
        w[w_bl>0] = (w_bl * self.get_is_weights())[w_bl>0] ##guard against any 0 * inf nonsense
        return w/w.sum()

    def get_ranges(self, *args, **kwargs):
        return self.base.get_ranges(*args, **kwargs)

    def get_sampler(self, *args, **kwargs):
        return self.base.get_sampler(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        return self.base.get_params(*args, **kwargs)

    def get_labels(self, *args, **kwargs):
        return self.base.get_labels(*args, **kwargs)

    def get_fiducial(self, *args, **kwargs):
        return self.base.get_fiducial(*args, **kwargs)
    
    # def get_1d_shift_integrated(self, param):
    #     base_mean, is_mean = self.base.get_mean(param), self.get_mean(param)
    #     vals = self.on_params(param)
    #     mi, ma = [base_mean, is_mean] if base_mean < is_mean else [is_mean, base_mean]
    #     return sum(self.base.get_weights()[(vals > mi)*(vals < ma)])

    def get_2d_shift(self, params):
        inv_cov = np.linalg.inv(self.base.get_MCSamples().cov(params))
        p = self.get_mean(params) - self.base.get_mean(params)
        return np.einsum('i,ij,j', p, inv_cov, p)
    
    def load_data(self, *args, **kwargs):
        nsample = self.base.nsample if hasattr(self.base, 'nsample') else 0
        super().load_data(*args, **kwargs, nsample=nsample)
        

def main():
    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('chain', help = 'base chain filename.')
    parser.add_argument('importance_weights', nargs='+', help = 'importance sampling weights filename.')
    parser.add_argument('output', help = 'output root.')

    parser.add_argument('--fig-format', dest = 'fig_format',
                    default = 'pdf', required = False,
                    help = 'export figures in specified format.')

    parser.add_argument('--triangle-plot', dest = 'triangle_plot', action='store_true',
                    help = 'generate triangle plots.')

    parser.add_argument('--base-plot', dest = 'base_plot', action='store_true',
                    help = 'include base chain in triangle plots.')

    parser.add_argument('--plot-weights', dest = 'plot_weights', action='store_true',
                    help = 'plot importance weights.')

    parser.add_argument('--stats', dest = 'stats', action='store_true',
                    help = 'compute importance sampling statistics.')

    parser.add_argument('--markdown-stats', dest = 'markdown_stats', action='store_true',
                    help = 'output short summary in markdown.')

    parser.add_argument('--summary', dest = 'summary', action='store_true',
                    help = 'do summary plot.')

    parser.add_argument('--shift-2d', dest = 'shift_2d', action='store_true',
                    help = "compute 2d bias using Shivam's code.")

    parser.add_argument('--all', dest = 'all', action='store_true',
                    help = 'same as --stats --triangle-plot --base-plot.')

    parser.add_argument('--classic-kde', dest = 'classic_kde', action='store_true',
                    help = 'Use a constant KDE kernel instead of getdist optimized kernel.')

    parser.add_argument('--extra-chains', dest = 'extra_chains', nargs='*', required = False,
                    help = 'Use this to include more chains in the plots.')

    parser.add_argument('--labels', dest = 'labels', nargs='*', required = False,
                    help = 'IS chain labels.')
    
    parser.add_argument('--boosted', dest = 'boosted',  action='store_true',
                    help = 'Load the baseline chain from the polychord output files rather than cosmosis output (useful if boost_posterior=T).')
    
    parser.add_argument('--base-weight', dest = 'base_weight',
                    default = 0, required = False,
                    help = 'define how the baseline weights will be determined (0: weight, 1: exp(log_weight)*old_weight, 2: old_weight.')
    
    args = parser.parse_args()

    if args.all:
        args.stats = True
        args.triangle_plot = True
        args.base_plot = True
        args.markdown_stats = True
        
    base_chain = Chain(args.chain, args.boosted, args.base_weight)
    is_chains = [ImportanceChain(iw_filename, base_chain) for i, iw_filename in enumerate(args.importance_weights)]
    extra_chains = [Chain(f) for f in args.extra_chains] if args.extra_chains else []

    N_IS = len(is_chains)

    if args.stats: #we should really probably change this to output in a more machine-readable format
        output_string = ''
        for chain in is_chains:
            ESS_base = chain.base.get_ESS_dict()
            ESS_IS = chain.get_ESS_dict()

            output_string += '\nFile: {}\n'.format(chain.filename)

            base_mean, base_std = chain.base.get_mean(params2plot), chain.base.get_std(params2plot)
            is_mean, is_std = chain.get_mean(params2plot), chain.get_std(params2plot)

            output_string += '\nBaseline mean ± std\n'
            for p,m,s in zip(params2plot, base_mean, base_std):
                output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, m, s)

            output_string += '\nImportance sampled mean ± std\n'
            for p,m,s in zip(params2plot, is_mean, is_std):
                output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, m, s)

            output_string += '\nDelta parameter/std ± 1/sqrt(ESS)\n'
            for p, bm, bs, im in zip(params2plot, base_mean, base_std, is_mean):
                output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, (im-bm)/bs, 1/np.sqrt(ESS_IS['Euclidean distance']))

            output_string += '\nDelta parameter (n-sigma)\n'
            for p, bm, im in zip(params2plot, base_mean, is_mean):
                vals = chain.base.data[p]
                mi, ma, si = [bm, im, 1] if bm < im else [im, bm, -1]
                pval_base = np.sum(chain.base.get_weights()[(vals > mi)*(vals < ma)])/np.sum(chain.base.get_weights())
                pval_is   = np.sum(chain.get_weights()[(vals > mi)*(vals < ma)])/np.sum(chain.get_weights())
                output_string += '\t{:<40} {:7n} (base), {:7n} (cont)\n'.format(p, si*np.sqrt(2)*sp.special.erfinv(2*pval_base),
                                                                                   si*np.sqrt(2)*sp.special.erfinv(2*pval_is))

            output_string += '\n2D bias\n'
            param_combinations = np.array(list())
            for p in itt.combinations(params2plot, 2):
                output_string += '\n\t{:<40}\n\t{:<40} {:7n}\n'.format(*p, chain.get_2d_shift(p))

            output_string += '\nDelta loglike (= -2*<delta chi^2>, want shifts of <~O(1))\n'
            dl = chain.get_dloglike_stats()
            output_string += '\tAverage: {:7n}\n'.format(dl[0])
            output_string += '\tRMS:     {:7n}\n'.format(dl[1])
            output_string += '\delta logZ:     {:7n}\n'.format(chain.get_delta_logz())

            output_string += '\nEffective sample sizes (rough rule of thumb is want ~ESS_IS/ESS_BL > ~0.1 and ESS_IS > ~100)\n'
            for key in ESS_base.keys():
                output_string += '\t{:<30}\t{:7n}/{:7n} = {:7n}\n'.format(key, ESS_IS[key], ESS_base[key], ESS_IS[key]/ESS_base[key])

            output_string += '\n\tTotal samples' + ' '*27 + '{}\n'.format(chain.N)

        with open(args.output + '_stats.txt', 'w') as f:
            f.write(output_string)

        print(output_string.replace('\n', '\r\n'))

    # Plot IS weights
    if args.plot_weights:
        for chain in is_chains:
            fig, ax = plot.subplots()
            ax.plot(chain.get_weights())
            # ax.plot(chain.base.get_weights())
            plot.savefig('{}_{}_weights.{}'.format(args.output, chain.name, args.fig_format))

    # Make triangle plot
    if args.triangle_plot:
        samples = []
        settings = {'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.4} if args.classic_kde else None
        if args.base_plot:
            samples.append(base_chain.get_MCSamples(settings=settings))
        samples.extend([is_chain.get_MCSamples(settings=settings) for is_chain in is_chains])
        samples.extend([c.get_MCSamples(settings=settings) for c in extra_chains])

        g = plots.getSubplotPlotter()

        g.triangle_plot(
            samples,
            params=params2plot,
            legend_labels=['Baseline'] + args.labels if args.base_plot else args.labels
        )

        # Write some stats to the triangle plot
        if len(is_chains) == 1:
            chain = is_chains[0]

            dl = chain.get_dloglike_stats()
            text_note_1 = 'Average delta loglike: {:n}\n'.format(dl[0])
            text_note_1 += 'RMS delta loglike: {:n}\n'.format(dl[1])

            text_note_1 += '\nEffective sample sizes\n'
            ESS_base = chain.base.get_ESS_dict()
            ESS_IS = chain.get_ESS_dict()
            for key in ESS_base.keys():
                text_note_1 += '{}: {:.0f}/{:.0f} = {:.0%}\n'.format(key, ESS_IS[key], ESS_base[key], ESS_IS[key]/ESS_base[key])

            text_note_1 += '\nTotal samples: {}\n'.format(chain.N)

#            g.add_text_left(text_note_1, ax=(0,0), x=1.03, y=0.35, fontsize=7)

            base_mean, base_std = chain.base.get_mean(params2plot), chain.base.get_std(params2plot)
            is_mean, is_std = chain.get_mean(params2plot), chain.get_std(params2plot)

            text_note_2 = 'Baseline mean ± std\n'
            for p,m,s in zip(param_to_label(params2plot), base_mean, base_std):
                text_note_2 += '${}$ {:3n} ± {:3n}\n'.format(p, m, s)

            text_note_2 += '\nImportance sampled mean ± std\n'
            for p,m,s in zip(param_to_label(params2plot), is_mean, is_std):
                text_note_2 += '${}$ {:3n} ± {:3n}\n'.format(p, m, s)

            text_note_2 += '\nDelta parameter/std\n'
            for p, bm, bs, im in zip(param_to_label(params2plot), base_mean, base_std, is_mean):
                text_note_2 += '${}$ {:3n}\n'.format(p, (im-bm)/bs)

            #g.add_text_left(text_note_2, ax=(1,1), x=1.03, y=0.35, fontsize=7)
            #g.add_text_left(text_note_2, ax=(1,1), x=1.03, y=0.5, fontsize=7)

        g.export(args.output + '_triangle.' + args.fig_format)

    if args.shift_2d:
        output_string = '\n2D shifts\n'
        for is_chain in is_chains:
            output_string += '\nFile: {}\n'.format(is_chain.filename)
            for p in itt.combinations(params2plot, 2):
                shifts = shivam_2d_bias.compute_2d_bias(base_chain.get_MCSamples(), is_chain.get_MCSamples(),
                        base_chain.get_fiducial(extra={'cosmological_parameters--sigma_8': 0.8430599612}),
                        *p, *param_to_latex(p), 0.01,
                        '{}_{}_{}_{}_2d_bias.{}'.format(args.output, is_chain.filename.split('/')[-1], *p, args.fig_format))

                output_string += '\n\t{}\n\t{}\n'.format(*p)
                for s in shifts.keys():
                    output_string += '\n\t\t{:<30}: {:3n}'.format(s, shifts[s])

            with open('{}_{}_{}_{}_2d_bias.txt'.format(args.output, is_chain.filename.split('/')[-1], *p), 'w') as f:
                f.write(output_string)

        print(output_string.replace('\n', '\r\n'))

    if args.markdown_stats:
        pairs = list(itt.combinations(params2plot, 2))
        output_string = '\pagenumbering{gobble}\n\n'
        output_string += '| | ' + '| '.join(['$\Delta {}/\sigma$'.format(param_to_label(p)) for p in params2plot]) + ' | ' + ('| '.join(['2D bias ${} \\times {}$'.format(*param_to_label(p)) for p in pairs]) if len(pairs) > 1 else '2D bias') + ' |\n'
        output_string += '| -: |' + ' :-: |'*(len(params2plot)+1 + len(pairs)) + '\n'
        for chain, label in zip(is_chains, args.labels):
            ESS_base = chain.base.get_ESS_dict()
            ESS_IS = chain.get_ESS_dict()

            base_mean, base_std = chain.base.get_mean(params2plot), chain.base.get_std(params2plot)
            is_mean, is_std = chain.get_mean(params2plot), chain.get_std(params2plot)

            biases_1d = (is_mean - base_mean)/base_std
            error_1d = 1/np.sqrt(ESS_IS['Euclidean distance'])
            biases_2d = ['${:.3f}$'.format(chain.get_2d_shift(p)) for p in pairs]

            output_string += '| {} | '.format(label) + ' |'.join(['${:+.3f} \\pm {:.3f}$'.format(b, error_1d) for b in biases_1d]) + ' | ' + ' |'.join(biases_2d) + ' |\n'

        with open(args.output + '_stats.md', 'w') as f:
            f.write(output_string)

    if args.summary:
        chains = [base_chain]
        chains.extend(is_chains)
        c = ChainConsumer()
        for i, chain in zip(range(len(chains)), chains):
            c.add_chain(chain.on_params(params2plot), parameters=param_to_latex(params2plot).tolist(), weights=chain.get_weights(), name='chain{}'.format(i))
        c.plotter.plot_summary(errorbar=True, truth='chain0', include_truth_chain=True, filename='{}_summary.{}'.format(args.output, args.fig_format))

if __name__ == "__main__":
    main()

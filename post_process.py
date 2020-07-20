#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plot
from getdist import MCSamples, plots
import argparse, configparser, copy

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
]

label_dict = {
    'cosmological_parameters--tau':  r'\tau',
    'cosmological_parameters--w':  r'w',
    'cosmological_parameters--wa':  r'w_a',
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
}

params2plot = [
     'cosmological_parameters--omega_m',
     'cosmological_parameters--sigma_8',
     'cosmological_parameters--s8',
#    'cosmological_parameters--w',
]

param_to_label = np.vectorize(lambda param: label_dict[param] if param in label_dict else param)

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

    def __init__(self, filename):
        self.filename = filename
        self.name = '.'.join(filename.split('/')[-1].split('.')[:-1])
        self.load_data()
        self.__add_extra()

    def load_data(self):
        data = []
        with open(self.filename) as f:
            labels = np.array(f.readline()[1:-1].lower().split())
            mask = ["data_vector" not in l for l in labels]
            for line in f.readlines():
                if '#' in line:
                    continue
                else:
                    data.append(np.array(line.split(), dtype=np.double)[mask])
        self.data = {labels[mask][i].lower(): col for i, col in enumerate(np.array(data).T)}
        self.N = len(self.data[labels[0]])
        return self.data

    def __add_extra(self):

        self.data['cosmological_parameters--s8'] = \
            self.data['cosmological_parameters--sigma_8']*(self.data['cosmological_parameters--omega_m']/0.3)**0.5

        self.data['cosmological_parameters--ommh2'] = \
            self.data['cosmological_parameters--omega_m']*self.data['cosmological_parameters--h0']**2

        self.data['cosmological_parameters--ombh2'] = \
            self.data['cosmological_parameters--omega_b']*self.data['cosmological_parameters--h0']**2

        self.data['cosmological_parameters--omch2'] = \
            self.data['cosmological_parameters--ommh2'] - self.data['cosmological_parameters--ombh2']

        self.data['cosmological_parameters--omega_c'] = \
            self.data['cosmological_parameters--omega_m'] - self.data['cosmological_parameters--omega_b']

    def get_sampler(self):
        """reads the sampler name from a given chain"""
        params = load_ini(self.filename, 'params')
        sampler = params.get('runtime', 'sampler')
        # print("Sampler is {}".format(sampler))
        return sampler

    def get_params(self):
        params = [l for l in self.data.keys() if l not in not_param]

        if len(params) > 0:
            return params
        else:
            raise Exception("No parameters found..")

    def get_labels(self):
        return param_to_label(self.get_params())

    def on_params(self):
        return np.array([self.data[l] for l in self.get_params()]).T

    def get_fiducial(self, filename=None):
        """loads range values from values.ini file or chain file"""

        values = load_ini(self.filename, ini='values' if filename is None else None)

        return {p:
            float((lambda x: x[1] if len(x) == 3 else x[0])(values.get(*p.split('--')).split())) \
            if values.has_option(*p.split('--')) \
            else None \
            for p in self.get_params()}

    def get_ranges(self, filename=None):
        """loads range values from values.ini file or chain file"""

        values = load_ini(self.filename, ini='values' if filename is None else None)

        return {p:
            (lambda x: [float(x[0]), float(x[2])] if len(x) == 3 else [None, None])(values.get(*p.split('--')).split()) \
            if values.has_option(*p.split('--')) \
            else [None, None] \
            for p in self.get_params()}

    def get_MCSamples(self, settings=None):

        return MCSamples(
            samples=self.on_params(),
            weights=self.get_weights(),
            #loglikes=self.get_likes(),

            ranges=self.get_ranges(),
            sampler='nested' if self.get_sampler() in ['multinest', 'polychord'] else 'mcmc',

            names=self.get_params(),
            labels=[l for l in self.get_labels()],
            settings=settings,
        )

    def get_weights(self):
        if 'weight' in self.data.keys():
            w = self.data['weight']
            return w/w.sum()
        else:
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
        """compute weighted average and rms of loglikelihood difference. should be <~ o(1).
        returns (dloglike, rms_dloglike)"""
        dloglike = np.average(self.data['old_like'] - self.data['new_like'], weights=self.data['old_weight'])
        rmsdloglike = np.average((self.data['old_like'] - self.data['new_like'])**2, weights=self.data['old_weight'])**0.5

        return dloglike, rmsdloglike

    def get_ESS_NW(self, weight_by_multiplicity=True):
        """compute and return effective sample size of is chain. 
        if is chain is identical to baseline, then just equals full sample size.
        insensitive to multiplicative scaling, i.e. if is chain shows all points exactly half as likely, will not show up in ess,
        so use mean_dloglike stat for that.
        (see e.g. https://arxiv.org/pdf/1602.03572.pdf or 
        http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html)"""
        #want stats on change to weights, but noisier if compute from new_weight/old_weight, so use e^dloglike directly.
        weight_ratio = np.exp(self.data['new_like'] - self.data['old_like'])
        nsamples = len(weight_ratio)
        if weight_by_multiplicity: 
            mult = self.data['old_weight']
        else:
            mult = np.ones_like(weight_ratio)
        normed_weights = weight_ratio / (np.average(weight_ratio, weights=mult)) #pulled out factor of nsamples
        return nsamples * 1./(np.average(normed_weights**2, weights=mult))


    def get_mean_err(self, params):
        return self.base.get_MCSamples().std(params)/self.get_ESS()**0.5

    def get_mean(self, params):
        return self.get_MCSamples().mean(params)

    def on_params(self):
        return self.base.on_params()

    def get_likes(self):
        return self.data['new_like']

    def get_is_weights(self):
        w = np.nan_to_num(np.exp(self.data['new_like'] - self.data['old_like']))
        return w

    def get_weights(self):
        w = np.nan_to_num(self.data['old_weight']*self.get_is_weights())

        return w/w.sum()

    def get_ranges(self, filename=None):
        return self.base.get_ranges(filename=filename)

    def get_sampler(self):
        return self.base.get_sampler()

    def get_params(self):
        return self.base.get_params()

    def get_labels(self):
        return self.base.get_labels()

    def on_params(self):
        return self.base.on_params()

    def get_fiducial(self, filename=None):
        return self.base.get_fiducial(filename=filename)

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

    parser.add_argument('--all', dest = 'all', action='store_true',
                    help = 'same as --stats --triangle-plot --base-plot.')

    parser.add_argument('--classic-kde', dest = 'classic_kde', action='store_true',
                    help = 'Use a constant KDE kernel instead of getdist optimized kernel.')

    parser.add_argument('--extra-chains', dest = 'extra_chains', nargs='*', required = False,
                    help = 'Use this to include more chains in the plots.')

    parser.add_argument('--legend-labels', dest = 'legend_labels', nargs='*', required = False,
                    help = 'Label chains in the triangle plot.')


    args = parser.parse_args()

    if args.all:
        args.stats = True
        args.triangle_plot = True
        args.base_plot = True
        args.plot_weights = True

    base_chain = Chain(args.chain)
    is_chains = [ImportanceChain(iw_filename, base_chain) for i, iw_filename in enumerate(args.importance_weights)]
    extra_chains = [Chain(f) for f in args.extra_chains] if args.extra_chains else []

    N_IS = len(is_chains)

    if args.stats:
        output_string = ''
        for chain in is_chains:
            output_string += '\nFile: {}\n'.format(chain.filename)

            base_mean, base_std = chain.base.get_mean(params2plot), chain.base.get_std(params2plot)
            is_mean, is_std = chain.get_mean(params2plot), chain.get_std(params2plot)

            output_string += '\nBaseline mean ± std\n'
            for p,m,s in zip(params2plot, base_mean, base_std):
                output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, m, s)

            output_string += '\nImportance sampled mean ± std\n'
            for p,m,s in zip(params2plot, is_mean, is_std):
                output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, m, s)

            output_string += '\nDelta parameter/std\n'
            for p, bm, bs, im in zip(params2plot, base_mean, base_std, is_mean):
                output_string += '\t{:<40} {:7n}\n'.format(p, (im-bm)/bs)

            output_string += '\nDelta loglike\n'
            dl = chain.get_dloglike_stats()
            output_string += '\tAverage: {:7n}\n'.format(dl[0])
            output_string += '\tRMS:     {:7n}\n'.format(dl[1])

            output_string += '\nEffective sample sizes\n'
            ESS_base = chain.base.get_ESS_dict()
            ESS_IS = chain.get_ESS_dict()
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
            ax.plot(chain.base.get_weights())
            plot.savefig('{}_{}_weights.{}'.format(args.output, chain.name, args.fig_format))

    # Make triangle plot
    if args.triangle_plot:
        samples = []
        settings = {'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.4} if args.classic_kde else None
        samples.extend([c.get_MCSamples(settings=settings) for c in extra_chains])
        if args.base_plot:
            samples.append(base_chain.get_MCSamples(settings=settings))
        samples.extend([is_chain.get_MCSamples(settings=settings) for is_chain in is_chains])

        g = plots.getSubplotPlotter()

        g.triangle_plot(
            samples,
            params=params2plot,
            legend_labels=args.legend_labels,
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

            g.add_text_left(text_note_1, ax=(0,0), x=1.03, y=0.35, fontsize=7)


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
            g.add_text_left(text_note_2, ax=(1,1), x=1.03, y=0.5, fontsize=7)


        g.export(args.output + '_triangle.' + args.fig_format)

if __name__ == "__main__":
    main()

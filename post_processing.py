#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plot
from chainconsumer import ChainConsumer
#matplotlib.rc('xtick', labelsize=16)
#matplotlib.rc('ytick', labelsize=16)
import argparse
import logging
logging.disable()

class Chain:
    """Description: Generic chain object"""

    def __init__(self, filename):
        self.not_param = [
                'like',
                'old_like',
                'delta_loglike',
                'new_like',
                'prior',
                'post',
                '2pt_like',
                'old_weight',
                'weight',
        ]

        self.label_dict = {
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
            'cosmological_parameters--h0':      r'h',
            'cosmological_parameters--omega_b': r'\Omega_b',
            'cosmological_parameters--n_s':     r'n_s',
            'cosmological_parameters--a_s':     r'A_s',
            'cosmological_parameters--omnuh2':  r'\Omega_{\nu}',
            'cosmological_parameters--sigma_8': r'\sigma_8',
            'cosmological_parameters--s8': r'S_8',
            'intrinsic_alignment_parameters--a': r'A_{IA}',
            'intrinsic_alignment_parameters--alpha': r'\alpha_{IA}',
            'bin_bias--b1': 'b_1',
            'bin_bias--b2': 'b_2',
            'bin_bias--b3': 'b_3',
            'bin_bias--b4': 'b_4',
            'bin_bias--b5': 'b_5',
            'shear_calibration_parameters--m1': 'm_1',
            'shear_calibration_parameters--m2': 'm_2',
            'shear_calibration_parameters--m3': 'm_3',
            'shear_calibration_parameters--m4': 'm_4',
            'lens_photoz_errors--bias_1': 'z^l_1',
            'lens_photoz_errors--bias_2': 'z^l_2',
            'lens_photoz_errors--bias_3': 'z^l_3',
            'lens_photoz_errors--bias_4': 'z^l_4',
            'lens_photoz_errors--bias_5': 'z^l_5',
            'wl_photoz_errors--bias_1': 'z^s_1',
            'wl_photoz_errors--bias_2': 'z^s_2',
            'wl_photoz_errors--bias_3': 'z^s_3',
            'wl_photoz_errors--bias_4': 'z^s_4',
        }
        self.__load(filename)

    def __load(self, filename):
        data = []
        with open(filename) as f:
            labels = np.array(f.readline()[1:-1].lower().split())
            mask = ["data_vector" not in l for l in labels]
            for line in f.readlines():
                if '#' in line:
                    continue
                else:
                    data.append(np.array(line.split(), dtype=np.double)[mask])
        self.data = {labels[mask][i]: col for i,col in enumerate(np.array(data).T)}
        return self.data

    def add_extra(self):
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

    def get_params(self):
        return [l for l in self.data.keys() if l not in self.not_param]

    def get_labels(self):
        return ['$'+self.label_dict[l]+'$' for l in self.get_params()]

    def on_params(self):
        return np.array([self.data[l] for l in self.get_params()]).T


class ImportanceChain(Chain):
    """Description: object to load the importance weights, plot and compute statistics.
       Should be initialized with reference to the respective baseline chain: ImportanceChain(base_chain)"""

    def __init__(self, filename, base_chain):
        super().__init__(filename)
        self.base = base_chain

    def get_diagnostics(self):
        """compute effective sample size and weighted average,rms of loglikelihood difference.
        returns (eff_nsample, mean_dloglike, rms_dloglike)"""

        assert self.data

        log_importance_weights = self.data['new_like'] - self.data['old_like']
        importance_weights = np.exp(log_importance_weights, dtype=np.float128)

        if(np.max(importance_weights) >= 1e158):
            pass

        mean_dloglike = np.average(log_importance_weights,    weights=self.data['old_weight'])
        rms_dloglike  = np.average(log_importance_weights**2, weights=self.data['old_weight'])**0.5

        mean_importance_weights_squared = np.average(importance_weights,    weights=self.data['old_weight'])**2
        rms_importance_weights_squared  = np.average(importance_weights**2, weights=self.data['old_weight'])

        eff_nsample_frac = mean_importance_weights_squared / rms_importance_weights_squared

        nsample = len(importance_weights)
        eff_nsample = nsample * eff_nsample_frac

        return eff_nsample, mean_dloglike, rms_dloglike

    def dist_from_baseline(self):
        self.__check_cc()

        params, cov_param = self.c.get_covariance(0)
        summ_0, summ_1 = self.c.get_summary()

        d_0 = np.array([summ_0[p][0] for p in self.base.get_labels()], dtype=np.double)
        d_1 = np.array([summ_1[p][0] for p in self.base.get_labels()], dtype=np.double)

        d = d_0 - d_1

        dist = np.sqrt(np.einsum('i,ij,j', d, np.linalg.inv(cov_param), d))

        return dist

    def __check_cc(self):
        if not hasattr(self, 'c'):
            self.__build_cc()

    def __build_cc(self):

        assert self.base and self.data

        self.c = ChainConsumer()

        weights = self.data['weight']
        mask = [w >= 1e-20 for w in weights]

        self.c.add_chain(self.base.on_params()[mask,:], parameters=self.base.get_labels(), name='Base',
                weights=self.base.data['weight'][mask] if 'weight' in self.base.data.keys() else None)

        self.c.add_chain(self.base.on_params()[mask,:], parameters=self.base.get_labels(), name='IS',
                weights=weights[mask]) 

        return self.c

    def plot(self, params2plot, output_dir, plot_base=False, kde=False):

        self.__check_cc()

        # Some plot configurations
        self.c.configure(linestyles="-", linewidths=1.0,
                    shade=False, shade_alpha=0.5, sigmas=[1,2], kde=kde,
                    label_font_size=20, tick_font_size=20, legend_color_text=True)

        fig = self.c.plotter.plot(parameters=params2plot)
        fig.set_size_inches(4.5 + fig.get_size_inches())
        fig.savefig(output_dir+"/triangle_plot.svg")

    def get_summary(self):
        """ returns the mean parameter values"""
        self.__check_cc()

        summary = self.c.analysis.get_summary(squeeze=False)
        return summary[1]

def main():
    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('chain', help = 'Base chain filename.')
    parser.add_argument('importance_weights', help = 'Importance sampling weights filename.')
    parser.add_argument('output', help = 'Output dir.')

    parser.add_argument('--burn-in', dest = 'burn',
                           default = 0, required = False,
                           help = 'Number of samples to burn-in.')

    parser.add_argument('--svg', dest = 'svg', action='store_true',
                           help = 'Export figures in svg format.')

    parser.add_argument('--triangle-plot', dest = 'triangle_plot', action='store_true',
                           help = 'Generate triangle plots.')

    parser.add_argument('--base-plot', dest = 'base_plot', action='store_true',
                           help = 'Include base chain in triangle plots.')

    parser.add_argument('--stats', dest = 'stats', action='store_true',
                           help = 'Compute importance sampling statistics.')

    parser.add_argument('--all', dest = 'all', action='store_true',
                           help = 'Same as --stats --triangle-plot --base-plot.')

    parser.add_argument('--kde', dest = 'kde', action='store_true',
                           help = 'Uses KDE smoothing in the triangle plot.')

    args = parser.parse_args()

    if args.all:
        args.stats = True
        args.triangle_plot = True
        args.base_plot = True

    base_chain = Chain(args.chain)
    importance = ImportanceChain(args.importance_weights, base_chain)
    
    #print(args.importance_weights, importance.get_summary()['$\\Omega_m$'][1], *importance.get_diagnostics())
    print(args.importance_weights, *importance.get_diagnostics())

    #importance.plot(['cosmological_parameters--omega_m'], args.output, plot_base=args.base_plot, kde=args.kde)
    #print(importance.stat_bias())

if __name__ == "__main__":
    main()

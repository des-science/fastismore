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

not_param = [
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
	'cosmological_parameters--h0':      r'h',
	'cosmological_parameters--omega_b': r'\Omega_b',
	'cosmological_parameters--n_s':     r'n_s',
	'cosmological_parameters--a_s':     r'A_s',
	'cosmological_parameters--omnuh2':  r'\Omega_{\nu}',
	'cosmological_parameters--sigma_8': r'\sigma_8',
	'cosmological_parameters--s8': r'S_8',
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
}

param_to_label = np.vectorize(lambda param: '${}$'.format(label_dict[param]))

def get_default_cc():
	cc = ChainConsumer()

	# Some plot configurations
	cc.configure(linestyles="-", linewidths=1.0,
		shade=False, shade_alpha=0.5, sigmas=[1,2], kde=kde,
		label_font_size=20, tick_font_size=20, legend_color_text=True)

def plot_cc(cc, params2plot):
	fig = cc.plotter.plot(parameters=param_to_label(params2plot))
	fig.set_size_inches(4.5 + fig.get_size_inches())
	return fig

class Chain:
	"""Description: Generic chain object"""

	def __init__(self, filename):
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
		return [l for l in self.data.keys() if l not in not_param]

	def get_labels(self):
		return param_to_label(self.get_params)

	def on_params(self):
		return np.array([self.data[l] for l in self.get_params()]).T

	def add_to_cc(self, cc):
		cc.add_chain(chain=self.on_params(), parameters=self.get_labels(), name='Baseline',
				weights=self.data['weight'] if 'weight' in self.data.keys() else None)

		return cc


class ImportanceChain(Chain):
	"""Description: object to load the importance weights, plot and compute statistics.
	   Should be initialized with reference to the respective baseline chain: ImportanceChain(base_chain)"""

	def __init__(self, filename, base_chain, name="IS"):
		super().__init__(filename)
		self.base = base_chain
		self.name = name

	def get_diagnostics(self):
		"""compute effective sample size and weighted average,rms of loglikelihood difference.
		returns (eff_nsample, mean_dloglike, rms_dloglike)"""

		return *self.get_dloglike_stats(), self.get_ESS(), self.dist_from_baseline()

	def get_bias_in_param(self, param):
		self.get_summary

	def get_dloglike_stats(self):
		"""compute weighted average and rms of loglikelihood difference. Should be <~ O(1).
		returns (dloglike, rms_dloglike)"""
		dloglike = -np.average(self.data['new_like'] - self.data['old_like'], weights=self.data['old_weight'])
		rmsdloglike = np.average((self.data['new_like'] - self.data['old_like'])**2, weights=self.data['old_weight'])**0.5
	
		return dloglike, rmsdloglike
	
	def get_ESS(ISdata, weight_by_multiplicity=True):
		"""compute and return effective sample size of IS chain. (author: Noah Weaverdyck)
		If IS chain is identical to baseline, then just equals full sample size.
		Insensitive to multiplicative scaling, i.e. if IS chain shows all points exactly half as likely, will not show up in ESS,
		so use mean_dloglike stat for that.
		(see e.g. https://arxiv.org/pdf/1602.03572.pdf or 
		http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html)"""
		#want stats on change to weights, but noisier if compute from new_weight/old_weight, so use e^dloglike directly.
		weight_ratio = np.exp(ISdata['new_like'] - ISdata['old_like'])
		Nsamples = len(weight_ratio)
		if weight_by_multiplicity: 
			mult = ISdata['old_weight']
		else:
			mult = np.ones_like(weight_ratio)
		normed_weights = weight_ratio / (np.average(weight_ratio, weights=mult)) #pulled out factor of Nsamples
		return Nsamples * 1./(np.average(normed_weights**2, weights=mult))

	def add_to_cc(self, cc, weight_threshold=1e-20):
		weights = self.data['weight']

		if weight_threshold:
			mask = [w >= weight_threshold for w in weights]
		else:
			mask = [True for w in weights]

		cc.add_chain(chain=self.base.on_params()[mask,:], parameters=self.base.get_labels(), name=self.name,
				weights=weights[mask])

		return cc

	def __build_cc(self, force_build=False):
		"""Some internal computations require a cc consumer object, so we build a basic one here."""

		if not force_build and hasattr(self, 'cc'):
			return self.cc

		assert self.base and self.data

		self.cc = get_default_cc()

		self.base.add_to_cc(self.cc)
		self.add_to_cc(self.cc)

		return self.cc

	def dist_from_baseline(self):
		"""Computes the distance between mean baseline and mean is in units of sigma."""
		self.__build_cc()

		params, cov_param = self.cc.get_covariance(0)
		summ_0, summ_1 = self.cc.get_summary()

		d_0 = np.array([summ_0[p][0] for p in self.base.get_labels()], dtype=np.double)
		d_1 = np.array([summ_1[p][0] for p in self.base.get_labels()], dtype=np.double)

		d = d_0 - d_1

		dist = np.sqrt(np.einsum('i,ij,j', d, np.linalg.inv(cov_param), d))

		return dist

	def plot(self, params2plot):
		"""Basic triangle plot with baseline and is contours superposed."""
		return plot_cc(self.__build_cc())

	def plot_weights(self, ax=None, plotbaseline=True, stats=True):
		if ax is None:
			f, ax = plt.subplots(figsize=(10,2))
		if plotbaseline:
			ax.plot(self.data['old_weight'], label='Baseline', zorder=2)
		ax.plot(self.data['weight'], label=self.name, alpha=0.5)
		ax.set_ylabel('weight', fontsize=20)
		ax.set_xlabel('Sample', fontsize=20)
		if stats:
			ax.text(0.01, 0.95, '({:.2g}, {:.2g}, {:.2g})'.format(get_ESS(self.data)/len(self.data['weight']), *get_dloglike_stats(self.data)),
					verticalalignment='top', horizontalalignment='left',
					transform=ax.transAxes, fontsize=16)
		ax.legend(loc=2, fontsize=20)

		return ax

	def get_summary(self):
		""" returns the mean parameter values"""
		self.__build_cc()

		return self.cc.analysis.get_summary(squeeze=False)[1]
		

	def get_summary(self, params):
		""" returns the mean parameter values"""
		self.__build_cc()

		labels = param_to_label(params)

		summary = self.cc.analysis.get_summary(squeeze=False)[1]
		return {key: summary[key] for key in summary.keys() if key in labels}

def main():
	parser = argparse.ArgumentParser(description = '')

	parser.add_argument('chain', help = 'Base chain filename.')
	parser.add_argument('importance_weights', nargs='+' help = 'Importance sampling weights filename.')
	parser.add_argument('output', help = 'Output root.')

	# parser.add_argument('--burn-in', dest = 'burn',
	# 					    default = 0, required = False,
	# 					    help = 'Number of samples to burn-in.')

	parser.add_argument('--fig-format', dest = 'fig_format',
						default = 'svg', required = False,
						help = 'Export figures in specified format.')

	parser.add_argument('--triangle-plot', dest = 'triangle_plot', action='store_true',
						help = 'Generate triangle plots.')

	parser.add_argument('--base-plot', dest = 'base_plot', action='store_true',
						help = 'Include base chain in triangle plots.')

	parser.add_argument('--plot-weights', dest = 'plot_weights', action='store_true',
						help = 'Plot importance weights.')

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
	is_chains = [ImportanceChain(iw_filename, base_chain, name='IS{}'.format(i)) for i, iw_filename in enumerate(args.importance_weights)]

	N_IS = len(is_chains)

	# Plot IS weights
	if args.plot_weights:
		f, axes = plt.subplots(N_IS, figsize=(10, 2*N_IS))
		for i,is_chain in enumerate(is_chains):
			is_chain.plot_weights(ax=axes[i] if N_IS > 1 else axes)
		f.savefig(args.output + '_weights.' + args.fig_format, bbox_inches='tight')

	# Make triangle plot
	if args.triangle_plot:
		cc = get_default_cc()
		
		base_chain.add_to_cc(cc)

		for is_chain in is_chains:
			is_chain.add_to_cc(cc)

		plot_cc(cc).savefig(args.output + '_triangle.' + args.fig_format)

		print(cc.analysis.get_summary(squeeze=False))

	if args.stats:
		for is_chain in is_chains:
			is_chain.get_diagnostics()
	
	#print(args.importance_weights, importance.get_summary()['$\\Omega_m$'][1], *importance.get_diagnostics())
	print(args.importance_weights, *importance.get_diagnostics())

	#importance.plot(['cosmological_parameters--omega_m'], args.output, plot_base=args.base_plot, kde=args.kde)
	#print(importance.stat_bias())

if __name__ == "__main__":
	main()

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plot
from getdist import MCSamples, plots
import argparse, configparser

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
}

params2plot = [
	 'cosmological_parameters--omega_m',
	 'cosmological_parameters--sigma_8',
]

param_to_label = np.vectorize(lambda param: label_dict[param])

def load_ini(filename, ini=None):
	"""loads given ini info from chain file. If ini=None, loads directly from file.ini"""
	values = configparser.ConfigParser(strict=False)

	if ini is None:
		values.read_file(filename)
	else:
		ini = ini.upper()
		with open(filename) as f:
			line = f.readline()
			lines=[]
			while("START_OF_{}".format(ini) not in line):
				line = f.readline()
			
			while("END_OF_{}".format(ini) not in line):
				line = f.readline()
				lines.append(line.replace('#', ''))
			values.read_string('\r'.join(lines[:-1]))
	return values

class Chain:
	"""Description: Generic chain object"""

	def __init__(self, filename):
		self.filename = filename
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
		self.data = {labels[mask][i].lower(): col for i,col in enumerate(np.array(data).T)}
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
		print("Sampler is {}".format(sampler))
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

	def get_MCSamples(self):
		
		return MCSamples(
			samples=self.on_params(),
			weights=self.data['weight'] if 'weight' in self.data.keys() else None,
			loglikes=self.data['like'],

			ranges=self.get_ranges(),
			sampler='nested' if self.get_sampler() in ['multinest', 'polychord'] else 'mcmc',

			names=self.get_params(),
			labels=[l for l in self.get_labels()],
		)


class ImportanceChain(Chain):
	"""Description: object to load the importance weights, plot and compute statistics.
	   Should be initialized with reference to the respective baseline chain: ImportanceChain(base_chain)"""

	def __init__(self, filename, base_chain):
		self.filename = filename
		self.base = base_chain
		self.load_data()

	def get_dloglike_stats(self):
		"""compute weighted average and rms of loglikelihood difference. Should be <~ O(1).
		returns (dloglike, rms_dloglike)"""
		dloglike = -np.average(self.data['new_like'] - self.data['old_like'], weights=self.data['old_weight'])
		rmsdloglike = np.average((self.data['new_like'] - self.data['old_like'])**2, weights=self.data['old_weight'])**0.5
	
		return dloglike, rmsdloglike
    
	def get_ESS_NW(self, weight_by_multiplicity=True):
	    """compute and return effective sample size of IS chain. 
	    If IS chain is identical to baseline, then just equals full sample size.
	    Insensitive to multiplicative scaling, i.e. if IS chain shows all points exactly half as likely, will not show up in ESS,
	    so use mean_dloglike stat for that.
	    (see e.g. https://arxiv.org/pdf/1602.03572.pdf or 
	    http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html)"""
	    #want stats on change to weights, but noisier if compute from new_weight/old_weight, so use e^dloglike directly.
	    weight_ratio = np.exp(self.data['new_like'] - self.data['old_like'])
	    Nsamples = len(weight_ratio)
	    if weight_by_multiplicity: 
	        mult = self.data['old_weight']
	    else:
	        mult = np.ones_like(weight_ratio)
	    normed_weights = weight_ratio / (np.average(weight_ratio, weights=mult)) #pulled out factor of Nsamples
	    return Nsamples * 1./(np.average(normed_weights**2, weights=mult))

	def get_ESS(self, weight_by_multiplicity=True):
		"""compute and return effective sample size."""
		# return self.get_MCSamples().getEffectiveSamples()

		w = self.data['weight']
		return w.sum()**2/(w**2).sum()

	def get_mean_err(self, params):
		return self.base.get_MCSamples().std(params)/self.get_ESS()**0.5

	def get_mean(self, params):
		return self.get_MCSamples().mean(params)

	def plot_weights(self, ax=None, plotbaseline=True):
		if ax is None:
			f, ax = plt.subplots(figsize=(10,2))
		if plotbaseline:
			ax.plot(self.data['old_weight'], label='Baseline', zorder=2)
		ax.plot(self.data['weight'], label=self.name, alpha=0.5)
		ax.set_ylabel('weight', fontsize=20)
		ax.set_xlabel('Sample', fontsize=20)
		ax.legend(loc=2, fontsize=20)

		return ax

	def get_MCSamples(self):
		if not hasattr(self, 'mcsamples'):
			self.mcsamples = MCSamples(
				samples=self.base.on_params(),
				weights=self.data['weight'],
				loglikes=self.data['new_like'],

				ranges=self.base.get_ranges(),
				sampler='nested' if self.base.get_sampler() in ['multinest', 'polychord'] else 'mcmc',

				names=self.base.get_params(),
				labels=[l for l in self.base.get_labels()],
			)

		return self.mcsamples

def main():
	parser = argparse.ArgumentParser(description = '')

	parser.add_argument('chain', help = 'Base chain filename.')
	parser.add_argument('importance_weights', nargs='+', help = 'Importance sampling weights filename.')
	parser.add_argument('output', help = 'Output root.')

	# parser.add_argument('--burn-in', dest = 'burn',
	# 					    default = 0, required = False,
	# 					    help = 'Number of samples to burn-in.')

	parser.add_argument('--fig-format', dest = 'fig_format',
						default = 'pdf', required = False,
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

	# parser.add_argument('--kde', dest = 'kde', action='store_true',
	# 					help = 'Uses KDE smoothing in the triangle plot.')

	args = parser.parse_args()

	if args.all:
		args.stats = True
		args.triangle_plot = True
		args.base_plot = True
		args.plot_weights = True

	base_chain = Chain(args.chain)
	is_chains = [ImportanceChain(iw_filename, base_chain) for i, iw_filename in enumerate(args.importance_weights)]

	N_IS = len(is_chains)

	# Plot IS weights
	if args.plot_weights:
		f, axes = plt.subplots(N_IS, figsize=(10, 2*N_IS))
		for i,is_chain in enumerate(is_chains):
			is_chain.plot_weights(ax=axes[i] if N_IS > 1 else axes)
		f.savefig(args.output + '_weights.' + args.fig_format, bbox_inches='tight')

	# Make triangle plot
	if args.triangle_plot:
		samples = []
		if args.base_plot:
			samples.append(base_chain.get_MCSamples())
		samples.extend([is_chain.get_MCSamples() for is_chain in is_chains])

		g = plots.getSubplotPlotter()

		g.triangle_plot(
			samples,
			params=params2plot,
			# markers=truth,
			# legend_labels=['DES Y3 3x2pt LCDM'],
			title_limit=1,
		)

		g.export(args.output + '_triangle.' + args.fig_format)

	if args.stats:
		means = '_mean\t'.join([p.split('--')[1] for p in params2plot]) + '_mean'
		errors = '_error\t'.join([p.split('--')[1] for p in params2plot]) + '_error'

		stats_out = "#filename\tess\t{}\t{}\r\n".format(means,errors)

		for is_chain in is_chains:
			means = '\t'.join([str(p) for p in is_chain.get_mean(params2plot)])
			errors = '\t'.join([str(p) for p in is_chain.get_mean_err(params2plot)])
			stats_out += '{}\t{}\t{}\t{}\r\n'.format(is_chain.filename, is_chain.get_ESS(), means, errors)
		print(stats_out)
				
if __name__ == "__main__":
	main()

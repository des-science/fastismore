# coding: utf-8

import numpy as np
import scipy as sp

import contourpy
import configparser
import functools

import getdist as gd
import getdist.plots

from . import parameters as fparams
from . import VERBOSE

__all__ = ['Chain', 'ImportanceChain']

class Chain:
    """Description: Generic chain object"""

    def __init__(self, data=None, weight_option="weight", getdist_settings=None):
        if data is not None:
            self.data = data
        self.weight_option = weight_option
        self.getdist_settings = getdist_settings

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data
        self.N = len(list(data.values())[0])

    @classmethod
    def from_file(cls, filename, boosted=False, weight_option="weight", getdist_settings=None, add_extra=False):
        """Initialize chain with given filename (full path). Set boosted=True
        if you want to load a boosted chain. If boosted_chain_fn is passed,
        use that, otherwise use default format/path for Y3 (i.e. a
        subdir /pcfiles/ with 'pc_' added and 'chain' dropped.

        weight_option:
            weight: use column "weight" as weight for chain.
            [Default and almost certainly what you want. Use subclass
            ImportanceChain for importance weights.]

            log_weight: use exp('log_weight')*'old_weight'
                                   as weight for chain

            old_weight: use 'old_weight' as weight for chain
        """
        self = cls(weight_option=weight_option, getdist_settings=getdist_settings)
        self.filename = filename
        self.weight_option = weight_option
        self.name = ".".join(filename.split("/")[-1].split(".")[:-1])
        self.chaindir = "/".join(filename.split("/")[:-1])
        # go to pcfiles subdir and drop 'chain' from beginning of name
        self.filename_boosted = self.chaindir + "/pcfiles/pc" + self.name[5:] + "_.txt"
        
        if boosted:
            self.load_boosted_data()
        else:
            self.load_data()
        if add_extra: fparams.add_extra(self.data)
        else: fparams.add_s8(self.data)

        return self

    def load_data(self, boosted=False, nsample=0):
        data = np.loadtxt(self.filename)
        with open(self.filename) as f:
            labels = np.array(f.readline()[1:-1].lower().split())
            mask = ["data_vector" not in l for l in labels]
            data = data[:, mask]  # filter data with mask
        if nsample != 0:
            self.nsample = data.shape[0]
        self.data = {
            labels[mask][i].lower(): col
            for i, col in enumerate(np.array(data)[-nsample:, :].T)
        }
        self.N = len(self.data[labels[0]])
        return self.data

    def load_boosted_data(self):
        """load and store data from the polychord output chain (which includes more
        samples if it was run with boost_posteriors=T) instead of the cosmosis chain.
        Retrieved using `self.filename_boosted`, which by default is at
        './pcfiles/pc_[cosmosis chain filename]_.txt' relative to the cosmosis chain
        referenced in self.filename.
        """

        data = []
        with open(self.filename) as f:  # get labels and mask from cosmosis chain output
            labels_cosmosis = np.array(f.readline()[1:-1].lower().split())
            # for size reasons, don't load data_vector terms
            mask_cosmosis = ["data_vector" not in l for l in labels_cosmosis]
            mask_cosmosis_params = [
                ("data_vector" not in l)
                and (l not in ["weight", "like", "prior", "post"])
                for l in labels_cosmosis
            ]
            mask_cosmosis_params_dv = [
                l not in ["weight", "like", "prior", "post"] for l in labels_cosmosis
            ]

            # order of columns in PC output files
            labels_pc = np.array(
                ["weight", "like"]
                + list(labels_cosmosis[mask_cosmosis_params_dv])
                + ["prior"]
            )
            mask_pc = ["data_vector" not in l for l in labels_pc]
        # boosted_data = np.genfromtxt(self.filename_boosted)[:,:] #array too large

        with open(self.filename_boosted) as f:
            for line in f.readlines():
                if "#" in line:
                    continue
                else:
                    data.append(np.array(line.split(), dtype=np.double)[mask_pc])

        self.data = {
            labels_pc[mask_pc][i].lower(): col for i, col in enumerate(np.array(data).T)
        }
        # PC originally stores -2*loglike, change to just loglike to match cosmosis
        self.data["like"] = -0.5 * self.data["like"]
        self.data["post"] = self.data["prior"] + self.data["like"]
        self.N = len(self.data[labels_pc[0]])

        return self.data

    def read_nsample(self):
        with open(self.filename, "r") as fi:
            for ln in fi:
                if ln.startswith("#nsample="):
                    nsamples = int(ln[9:])
        return nsamples

    def get_sampler(self):
        """reads the sampler name from a given chain"""
        sampler = self.params().get("runtime", "sampler")
        if VERBOSE:
            print("Sampler is {}".format(sampler))
        return sampler

    def has_param(self, param):
        return param in self.data.keys() or (hasattr(self, 'base') and param in self.base.data.keys())

    def get_params(self):
        params = [l for l in self.data.keys() if l not in fparams.not_param]

        if len(params) > 0:
            return params
        else:
            raise Exception("No parameters found..")

    def get_labels(self, params=None):
        if params == None:
            params = self.get_params()
        return fparams.param_to_label(params)

    def on_params(self, params=None):
        if params == None:
            params = self.get_params()
        if type(params) == str:
            params = [params]
        return np.array([self.data[l] for l in params]).T

    def get_fiducial(self, filename=None, extra=None):
        """loads range values from values.ini file or chain file"""

        fiducial = {
            p: (
                float(
                    (lambda x: x[1] if len(x) == 3 else x[0])(
                        self.values().get(*p.split("--")).split()
                    )
                )
                if self.values().has_option(*p.split("--"))
                else None
            )
            for p in self.get_params()
        }

        return fparams.add_extra(fiducial, extra)

    def load_ini(self, ini="params", filename=None):
        """loads given ini info from chain file. If ini=None, loads directly from file.ini"""
        values = configparser.ConfigParser(strict=False)

        if filename is None:
            try:
                filename = self.filename
            except:
                raise Exception("Can't determine ini file.")

        ini = ini.upper()
        with open(filename) as f:
            line = f.readline()
            lines = []

            if VERBOSE:
                print("Looking for START_OF_{} in file {}".format(ini, filename))

            while "START_OF_{}".format(ini) not in line:
                line = f.readline()
                if line == "":
                    raise Exception(
                        "START_OF_{} not found in file {}.".format(ini, filename)
                    )

            while "END_OF_{}".format(ini) not in line:
                line = f.readline()
                lines.append(line.replace("#", ""))
                if line == "":
                    raise Exception(
                        "END_OF_{} not found in file {}.".format(ini, filename)
                    )

        values.read_string("\r".join(lines[:-1]))

        return values

    def load_params_from_chain(self, filename):
        self._params = self.load_ini(ini="params", filename=filename)
        self._values = self.load_ini(ini="values", filename=filename)

    def params(self):
        if not hasattr(self, "_params"):
            self._params = self.load_ini("params")
        return self._params

    def values(self):
        if not hasattr(self, "_values"):
            self._values = self.load_ini("values")
        return self._values

    def get_ranges(self, filename=None, params=None):
        """loads range values from values.ini file or chain file"""

        if hasattr(self, 'ranges'):
            return self.ranges

        if params == None:
            params = self.get_params()

        def gr(x):
            if len(x) == 3:
                return [float(x[0]), float(x[2])]
            elif len(x) == 2:
                return [float(x[0]), float(x[1])]
            else:
                return [None, None]

        self.ranges = {
            p: (
                gr(self.values().get(*p.split("--")).split()) \
                if self.values().has_option(*p.split("--")) \
                else [None, None]
            )
            for p in params
        }

        return self.ranges

    @functools.cache
    def get_MCSamples(self, params=None):

        if params == None:
            params = self.get_params()

        mc_params = {
            'samples': self.on_params(params=params),
            'weights': self.get_weights(),
            # loglikes=self.get_likes(),
            'names': params,
            'labels': [l for l in self.get_labels(params=params)],
            'settings': self.getdist_settings,
        }
        try:
            mc_params.update({
                'ranges':  self.get_ranges(params=params),
                'sampler': (
                    "nested" if self.get_sampler() in ["multinest", "nautilus", "polychord"] else "mcmc"
                ),
            })
            
        except:
            print('Ranges not found.')

        return gd.MCSamples(**mc_params)

    def get_weights(self):
        if self.weight_option == "weight" and "weight" in self.data.keys():
            if VERBOSE:
                print('Using column "weight" as weight for baseline chain.')
            w = self.data["weight"]
            return w / w.sum()
        elif (
            self.weight_option == "log_weight"
            and "log_weight" in self.data.keys()
            and "old_weight" in self.data.keys()
        ):
            if VERBOSE:
                print(
                    'Using "exp(log_weight)*old_weight" as weight for baseline chain.'
                )
            w = self.data["old_weight"]
            return w / w.sum()
        elif (
            self.weight_option == "log_weight"
            and "log_weight" in self.data.keys()
            and "old_weight" not in self.data.keys()
        ):
            if VERBOSE:
                print(
                    'Using "exp(log_weight)" as weight for nautilus chain.'
                )
            w = np.exp(self.data["log_weight"])
            return w / w.sum()
        elif self.weight_option == "old_weight" and "old_weight" in self.data.keys():
            if VERBOSE:
                print('Using column "old_weight" as weight for baseline chain.')
            w = self.data["old_weight"]
            return w / w.sum()
        
        raise Exception(f"No weight criteria satisfied. weight_option = {self.weight_option}")

    def get_likes(self):
        return self.data["like"]

    def get_mean_err(self, params):
        return self.get_MCSamples().std(params) / self.get_ESS() ** 0.5

    def get_std(self, params):
        return self.get_MCSamples().std(params)

    def get_mean(self, params):
        return self.get_MCSamples().mean(params)

    def thin(self, target_size):

        weights = self.get_weights()
        mask = np.random.choice(len(weights), size=target_size, replace=True, p=weights)
        
        # filter the chain:
        print(f'Keeping {target_size} out of {len(weight)} samples.')
        for key, value in self._data.items():
            self._data[key] = value[mask]
        self.N = target_size

    def bernoulli_thin(self, temperature=1, nrepeats=1):

        log_weight = np.log(self.get_weights())
        new_weight = np.exp((1. - temperature) * log_weight)
        log_weight = temperature*(log_weight - np.amax(log_weight))

        # do the trial:
        mask = np.zeros(len(log_weight)).astype(bool)
        sample_repeat = np.zeros(len(log_weight)).astype(int)
        for i in range(nrepeats):
            temp = np.random.binomial(1, np.exp(log_weight))
            sample_repeat += temp.astype(int)
            mask = np.logical_or(mask, temp.astype(bool))
        new_weight = sample_repeat*new_weight

        # filter the chain:
        print(f'Keeping {mask.sum()} out of {len(mask)} samples.')
        for key, value in self._data.items():
            self._data[key] = value[mask]
        self.N = mask.sum()

    def find_sigma_1d(self, param, ref=0, **kwargs):
        opt_kwargs={'bracket': (1e-3, 2)}
        opt_kwargs.update(kwargs)
        
        peak = self.get_peak_1d(param)
        if ref == peak:
            return 0.
        elif ref > peak:
            func_opt = lambda sigma: self.get_bounds(param, sigma=sigma)[1] - ref
        elif ref < peak:
            func_opt = lambda sigma: self.get_bounds(param, sigma=sigma)[0] - ref
        result = sp.optimize.root_scalar(func_opt, **opt_kwargs).root
        if ref > peak:
            result *= -1
        return result
    
    def get_bounds(self, param, sigma=1, method='peak_isolike', maxlike=False):
        if method == 'peak_isolike':
            return self.get_bounds_peak_isolike(param, sigma=sigma, maxlike=maxlike)
        else:
            raise Exception('Method not implemented.')
    
    @functools.cache
    def get_bounds_peak_isolike(self, param, sigma=1, maxlike=False):
        lower, upper, _, _ = self.get_MCSamples().get1DDensity(param).getLimits(sp.special.erf(sigma/np.sqrt(2)))
        if maxlike:
            return lower, self.get_peak_1d(param), upper
        else:
            return lower, upper

    @functools.cache
    def get_bounds_peak_isolike_legacy(self, param, sigma=1):
        print(f'Getting {sigma}σ bounds for {param}')
        pdf = self.get_1d_kde(param)

        x_grid   = np.linspace(pdf.dataset.min(), pdf.dataset.max(), 10000)
        pdf_grid = pdf(x_grid)

        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]
        cdf = sp.interpolate.interp1d(x_grid, cdf_grid, fill_value=(0, 1), bounds_error=False)

        peak_grid = np.argmax(pdf_grid)

        left_ipdf  = sp.interpolate.interp1d(
            (pdf_grid[:peak_grid][::-1]),
            x_grid[:peak_grid][::-1],
            bounds_error=False
        )
            
        right_ipdf = sp.interpolate.interp1d(
            (pdf_grid[peak_grid:]),
            x_grid[peak_grid:],
            bounds_error=False
        )

        ipdf_range = \
            max((pdf_grid[:peak_grid]).min(), (pdf_grid[peak_grid:]).min()), \
            min((pdf_grid[:peak_grid]).max(), (pdf_grid[peak_grid:]).max())

        cdf_goal = sp.special.erf(sigma / np.sqrt(2))

        opt_func = lambda like: np.abs(cdf(right_ipdf(like)) - cdf(left_ipdf(like)) - cdf_goal)
        opt_result = sp.optimize.minimize_scalar(opt_func, bounds=ipdf_range)

        print(f'Bounds converged')

        return left_ipdf(opt_result.x), x_grid[peak_grid], right_ipdf(opt_result.x)

    def get_bounds_mean_symmetric(self, param, sigma=1):
        mean = self.get_mean([param])[0]
        vals = self.on_params([param])[:, 0]
        weights = self.get_weights()

        i = np.argsort(vals)

        iright = i[vals[i] > mean]
        ileft  = i[vals[i] < mean][::-1]

        right = vals[iright[np.argmax(np.cumsum(weights[iright]) > sp.special.erf(sigma / np.sqrt(2)) / 2)]]
        left  = vals[ileft [np.argmax(np.cumsum(weights[ileft])  > sp.special.erf(sigma / np.sqrt(2)) / 2)]]

        return left, mean, right

    
    def get_bounds_quantiles(self, param, sigma=1):
        p = sp.special.erf(sigma/np.sqrt(2))
        quantiles = (1-p)/2, 0.5, (1+p)/2

        vals = self.on_params([param])[:, 0]
        weights = self.get_weights()

        i = np.argsort(vals)

        cdf = np.cumsum(weights[i])
        cdf /= cdf[-1]

        return vals[i][[np.searchsorted(cdf, q) for q in quantiles]]

    def get_ESS(self):
        """compute and return effective sample size."""

        w = self.get_weights()
        return 1.0 / (w**2).sum()

    def get_ESS_dict(self):
        """compute and return effective sample size."""
        if not hasattr(self, "ESS_dict"):

            w = self.get_weights()
            N = len(w)

            self.ESS_dict = {
                # overestimates
                "Euclidean distance": 1 / (w**2).sum(),

                 # underestimates; best when bias is large
                "Inverse max weight": 1 / np.max(w),

                 # best when bias and N are small
                "Gini coefficient": -2 * np.sum(np.arange(1, N + 1) * np.sort(w)) + 2 * N + 1,

                "Square root sum": np.sum(np.sqrt(w)) ** 2,
                "Peak integrated": -N * np.sum(w[w >= 1 / N]) + np.sum(w >= 1 / N) + N,
                "Shannon entropy": 2 ** (-np.sum(w[w > 0] * np.log2(w[w > 0]))),
                # Not stable
                # 'Maximum': N + 1 - N*np.max(w),
                # 'Peak count': np.sum(w>=1/N),
                # 'Minimum': N*(N - 1)*np.min(w) + 1,
                # 'Inverse minimum': 1/((1-N)*np.min(w) + 1),
                # 'Entropy': N - 1/np.log2(N)*(-np.sum(w[w>0]*np.log2(w[w>0]))),
                # 'Inverse entropy': -N*np.log2(N)/(-N*np.log2(N) + (N - 1)*(-np.sum(w[w>0]*np.log2(w[w>0])))),
            }
        return self.ESS_dict

    @functools.cache
    def get_1d_kde(self, param):
        kde = sp.stats.gaussian_kde(self.on_params([param])[:,0], weights=self.get_weights())
        return kde

    def get_density_1d(self, param):
        return self.get_MCSamples().get1DDensity(param)

    def get_sigma_1d(self, param, a, b):
        left, right, signal = a, b, 1
        if left > right:
            left, right = right, left
            signal = -1
        try:
            vals = self.data[param]
        except KeyError:
            vals = self.base.data[param]
            
        mask = (vals > left) * (vals < right)

        pval = np.sum(self.get_weights()[mask]) / np.sum(
            self.get_weights()
        )

        return signal * np.sqrt(2) * sp.special.erfinv(2 * pval)

    @functools.cache
    def get_density_grid(self, param1, param2):
        return gd.plots.MCSampleAnalysis([]).get_density_grid(
            self.get_MCSamples(),
            *[gd.paramnames.ParamInfo(name=p) for p in [param1, param2]],
        )

    def get_contour_vertices(self, sigma, param1, param2):
        density = self.get_density_grid(param1, param2)
        contour_levels = density.getContourLevels(sp.special.erf(np.atleast_1d(sigma) / np.sqrt(2)))
        contours = []
        for c in np.atleast_1d(contour_levels):
            contours += contourpy.contour_generator(density.x, density.y, density.P).lines(c)
        return contours

    def get_peak_2d(self, param1, param2):
        density = self.get_density_grid(param1, param2)
        j, i = np.unravel_index(np.argmax(density.P), density.P.shape)
        return density.x[i], density.y[j]

    def get_peak_1d(self, param):
        density = self.get_density_1d(param)
        #return sp.optimize.minimize_scalar(lambda x: -density.Prob(x)).x
        f = lambda x: -density.Prob(x)
        result = sp.optimize.minimize_scalar(f, bounds=(density.x.min(), density.x.max()), method='bounded') 
        return result.x

        # density = self.get_1d_kde(param)
        # return sp.optimize.minimize_scalar(lambda x: -density(x),
        #     bounds=(density.dataset.min(), density.dataset.max())).x

    def find_sigma_of_point(self, point, param1, param2):
        # finds the distance between point and closest vertex in nsigma contour
        def distance_to_nsigma_contour(sigma):
            return np.min(
                # distances between point and all nsigma contour vertices
                np.linalg.norm(np.concatenate(self.get_contour_vertices(sigma, param1, param2)) - point,
                    axis=1,
                )
            )

        # returns nsigma of contour that is closest to point
        return sp.optimize.minimize_scalar(
            distance_to_nsigma_contour, bounds=[1e-4, 8]
        ).x

    def get_param_differences(self, baseline=None, params=None, min_weight=0, boost=None):

        if params is None:
            params = self.get_params()
        
        if baseline is None:
            baseline = self.base

        if boost is None: # compute all pairs
            values = self.on_params(params)[np.newaxis,:,:] - baseline.on_params(params)[:,np.newaxis,:]
            values = values.reshape(-1, values.shape[-1])

            weights = self.get_weights() * baseline.get_weights()[:, np.newaxis]
            weights = weights.ravel()
        else:
            import tensiometer.mcmc_tension
            diff_gd = tensiometer.mcmc_tension.parameter_diff_weighted_samples(
                self.get_MCSamples(params),
                baseline.get_MCSamples(params),
                boost=boost
            )
        
            values = diff_gd.samples
            weights = diff_gd.weights
        # else: # iterate with different lags between sample sets
        #     lotta_samples = self     if self.N > baseline.N else baseline
        #     fewer_samples = baseline if self.N > baseline.N else self

        #     values  = np.empty((lotta_samples.N*iterations, len(params)))
        #     weights = np.empty( lotta_samples.N*iterations)

        #     for lag in range(iterations):
        #         start = int(lag/iterations*fewer_samples.N)
        #         indices = range(start, start+lotta_samples.N)
                
        #         values[lag*lotta_samples.N:(lag+1)*lotta_samples.N, :] = \
        #             lotta_samples.on_params(params) - np.take(fewer_samples.on_params(params), indices, axis=0, mode='wrap')
                
        #         weights[lag*lotta_samples.N:(lag+1)*lotta_samples.N] = \
        #             lotta_samples.get_weights() * np.take(fewer_samples.get_weights(), indices, mode='wrap')

        # Removing low-weight samples
        if min_weight > 0:
            max_weight = np.max(weights)
            mask = weights > min_weight*max_weight
            values = values[mask]
            weights = weights[mask]

        # Var(x - y) = Var(x) + Var(y) ~ 2*Var(x)
        # Here we remove that factor of 2
        average = np.average(values, weights=weights, axis=0) 
        values -= average # center distribution
        values /= np.sqrt(2) # shring it
        values += average # shift it back to original position
        
        data = {p:c for p,c in zip(params, values.T)}
        data['weight'] = weights
        
        differences = Chain(data=data)
            
        differences.filename = baseline.filename
        
        differences._params = baseline._params
        differences._values = baseline._values

        differences.truth = {p:0 for p in params}
        differences.ranges = None

        return differences

    def get_1d_shift(self, param, baseline=None):
        if baseline is None:
            baseline = self.base
        a, b = baseline.get_mean(param), self.get_mean(param)
        return baseline.get_sigma_1d(param, a, b), self.get_sigma_1d(param, a, b)

    def get_2d_shift_peak(self, param1, param2, baseline=None, base_posterior=True):
        if baseline is None:
            baseline = self.base

        posterior, peak = self, baseline

        if base_posterior:
            posterior, peak = peak, posterior
            
        return posterior.find_sigma_of_point(
            peak.get_peak_2d(param1, param2), param1, param2
        )

    def get_2d_shift_gaussian(self, param1, param2, baseline=None, base_posterior=True, mean=True):
        if baseline is None:
            baseline = self.base

        inv_cov = np.linalg.inv(
            (baseline if base_posterior else self).get_MCSamples().cov([param1, param2])
        )
        if mean:
            p = self.get_mean([param1, param2]) - baseline.get_mean([param1, param2])
        else:
            p = np.array(self.get_peak_2d(param1, param2)) - \
                np.array(baseline.get_peak_2d(param1, param2))

        return np.einsum("i,ij,j", p, inv_cov, p)

    def get_2d_shift_mean(self, param1, param2, base_posterior=True):
        posterior = baseline if base_posterior else self
        contaminated = self.get_mean([param1, param2])
        baseline = baseline.get_mean([param1, param2])

        sigma_base = posterior.find_sigma_of_point(baseline, param1, param2)
        sigma_cont = posterior.find_sigma_of_point(contaminated, param1, param2)

        return np.sqrt(2) * sp.special.erfinv(sp.special.erf(sigma_cont/np.sqrt(2)) - sp.special.erf(sigma_base/np.sqrt(2)))


    def get_2d_shift(self, param1, param2, mode="max", gaussian=False, base_posterior=True):
        if mode not in ["mean", "max"]:
            raise Exception(f"Mode has to be 'mean' or 'max'. Given value was {mode}")

        if gaussian:
            return self.get_2d_shift_gaussian(
                param1, param2, base_posterior, mode == "mean"
            )
        elif mode == "mean":
            return self.get_2d_shift_mean(param1, param2, base_posterior)
        elif mode == "max":
            return self.get_2d_shift_peak(param1, param2, base_posterior)

    def get_jaccard_index(self, sigma, param1, param2, baseline=None):
        """Returns the Jaccard index between baseline and contaminated n-sigma contours.
        Jaccard index is the ratio intersection/union of the two sets. Its value is 0
        when there is no overlap and 1 when there's complete overlap."""
        import shapely as shp

        if baseline is None:
            baseline = self.base

        poly_baseline = shp.geometry.Polygon(
            baseline.get_contour_vertices(sigma, param1, param2)[0]
        )
        poly_contaminated = shp.geometry.Polygon(
            self.get_contour_vertices(sigma, param1, param2)[0]
        )

        intersection = poly_baseline.intersection(poly_contaminated)
        union = poly_baseline.union(poly_contaminated)
        return intersection.area / union.area

class ImportanceChain(Chain):
    """Description: object to load the importance weights, plot and compute statistics.
    Should be initialized with reference to the respective baseline chain: ImportanceChain(base_chain)
    """

    def __init__(self, base_chain, data=None, getdist_settings=None):
        self.base = base_chain
        if data is not None:
            self.data = data
        self.getdist_settings = getdist_settings

    @classmethod
    def from_file(cls, filename, base_chain, getdist_settings=None):
        self = cls(base_chain, getdist_settings=getdist_settings)
        self.filename = filename
        self.name = ".".join(filename.split("/")[-1].split(".")[:-1])
        self.load_data()
        return self

    def get_dloglike_stats(self):
        """compute weighted average and rms of loglikelihood difference from baseline to IS chain.
        Deviance is -2*loglike; for Gaussian likelihood, dloglike = -0.5 * <delta chi^2>.
        RMS is included for back-compatibility. It can capture some differences that dloglike misses,
        but these are largely captured by ESS, so dloglike and ESS should work as primary quality
        statistics. Should be <~ o(1).
        returns (dloglike, rms_dloglike)"""
        dloglike = np.average(
            self.data["new_like"] - self.data["old_like"],
            weights=self.data["old_weight"],
        )
        rmsdloglike = (
            np.average(
                (self.data["new_like"] - self.data["old_like"]) ** 2,
                weights=self.data["old_weight"],
            )**0.5
        )

        return dloglike, rmsdloglike

    def get_ESS_NW(self, weight_by_multiplicity=True):
        """compute and return effective sample size of is chain.
        Is a little more correct (?) than using euclidean ESS of final weights in how it treats the
        fact that initial chain is weighted, so we shouldn't be able to get a higher effective sample
        size by adding additional weights. Difference is small in practice. If is chain is identical
        to baseline, then just equals full sample size. insensitive to multiplicative scaling,
        i.e. if IS chain shows all points exactly half as likely, will not show up in ess, use
        mean_dloglike stat for that. (see e.g. https://arxiv.org/pdf/1602.03572.pdf or
        http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html)
        """
        # Noisier if compute from new_weight/old_weight, so use e^dloglike directly.
        weight_ratio = np.exp(self.data["new_like"] - self.data["old_like"])
        nsamples = len(weight_ratio)
        if weight_by_multiplicity:
            mult = self.data["old_weight"]
        else:
            mult = np.ones_like(weight_ratio)
        # pulled out factor of nsamples
        normed_weights = weight_ratio / (np.average(weight_ratio, weights=mult))
        return nsamples * 1.0 / (np.average(normed_weights**2, weights=mult))

    def get_delta_logz(self):
        """get estimate on shift of evidence. Note that only uses posterior points; won't account
        for contributions from changes in  volume of likelihood shells"""
        w_is = self.get_is_weights()
        w_bl = self.base.get_weights()
        return np.log(np.average(w_is[w_bl > 0], weights=w_bl[w_bl > 0]))

    def get_mean_err(self, params):
        return self.base.get_MCSamples().std(params) / self.get_ESS() ** 0.5

    def on_params(self, *args, **kwargs):
        return self.base.on_params(*args, **kwargs)

    def on_params(self, params=None):
        if params == None:
            params = self.get_params()
        # data = self.base.data
        data = fparams.add_extra(self.base.data)
        # data.update(self.data)

        return np.array([data[l] for l in params]).T

    def get_likes(self):
        return self.data["new_like"]

    def get_is_weights(self, regularize=True):
        """If regularize=True (default), divide by maximum likelihood difference before
        computing weights. Cancels when normalize weights, and helps with overflow when
        large offset in loglikelihoods."""

        nonzero = self.base.get_weights() != 0
        likediff = self.data["new_like"] - self.data["old_like"]
        if regularize == True:
            maxdiff = np.max(likediff[nonzero])
        w = np.nan_to_num(np.exp(likediff - maxdiff))
        if "extra_is_weight" in self.data.keys():
            if VERBOSE:
                print("Using extra IS weight.")
            w *= self.data["extra_is_weight"]
        return w

    def get_weights(self):
        # w = np.nan_to_num(self.data['old_weight']*self.get_is_weights())
        if VERBOSE:
            print("WARNING: getting IS weights.")
        w_bl = self.base.get_weights()
        w = np.zeros_like(w_bl)
        # guard against any 0 * inf nonsense
        w[w_bl > 0] = (w_bl * self.get_is_weights())[w_bl > 0]
        return w / w.sum()

    def get_ranges(self, *args, **kwargs):
        return self.ranges if hasattr(self, 'ranges') else self.base.get_ranges(*args, **kwargs)

    def get_sampler(self, *args, **kwargs):
        return self.base.get_sampler(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        return self.base.get_params(*args, **kwargs)

    def get_labels(self, *args, **kwargs):
        return self.base.get_labels(*args, **kwargs)

    def get_fiducial(self, *args, **kwargs):
        return self.base.get_fiducial(*args, **kwargs)

    def load_data(self, *args, **kwargs):
        nsample = self.base.nsample if hasattr(self.base, "nsample") else 0
        super().load_data(*args, **kwargs, nsample=nsample)
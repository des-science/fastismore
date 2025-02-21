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

    def __init__(self, filename, boosted=False, weight_option="weight"):
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
        fparams.add_extra(self.data)

    def load_data(self, boosted=False, nsample=0):
        data = []

        with open(self.filename) as f:
            labels = np.array(f.readline()[1:-1].lower().split())
            mask = ["data_vector" not in l for l in labels]
            for line in f.readlines():
                if "#nsample" in line:
                    nsample = int(line.replace("#nsample=", ""))
                    if VERBOSE:
                        print(f"Found nsample = {nsample}")
                elif "#" in line:
                    continue
                else:
                    data.append(np.array(line.split(), dtype=np.double)[mask])
        if nsample != 0:
            self.nsample = nsample
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

    def load_ini(self, ini="params"):
        """loads given ini info from chain file. If ini=None, loads directly from file.ini"""
        values = configparser.ConfigParser(strict=False)

        ini = ini.upper()
        with open(self.filename) as f:
            line = f.readline()
            lines = []

            if VERBOSE:
                print("Looking for START_OF_{} in file {}".format(ini, self.filename))

            while "START_OF_{}".format(ini) not in line:
                line = f.readline()
                if line == "":
                    raise Exception(
                        "START_OF_{} not found in file {}.".format(ini, self.filename)
                    )

            while "END_OF_{}".format(ini) not in line:
                line = f.readline()
                lines.append(line.replace("#", ""))
                if line == "":
                    raise Exception(
                        "END_OF_{} not found in file {}.".format(ini, self.filename)
                    )

        values.read_string("\r".join(lines[:-1]))

        return values

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

        if params == None:
            params = self.get_params()

        self.ranges = {
            p: (
                (lambda x: [float(x[0]), float(x[2])] if len(x) == 3 else [None, None])(
                    self.values().get(*p.split("--")).split()
                )
                if self.values().has_option(*p.split("--"))
                else [None, None]
            )
            for p in params
        }

        return self.ranges

    @functools.cache
    def get_MCSamples(self, settings=None, params=None):

        if params == None:
            params = self.get_params()

        return gd.MCSamples(
            samples=self.on_params(params=params),
            weights=self.get_weights(),
            # loglikes=self.get_likes(),
            ranges=self.get_ranges(params=params),
            sampler=(
                "nested" if self.get_sampler() in ["multinest", "polychord"] else "mcmc"
            ),
            names=params,
            labels=[l for l in self.get_labels(params=params)],
            settings=settings,
        )

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

    def get_bounds(self, param, sigma=1):
        mean = self.get_mean([param])[0]
        vals = self.on_params([param])[:, 0]
        weights = self.get_weights()

        i = np.argsort(vals)

        iright = i[vals[i] > mean]
        ileft = i[vals[i] < mean][::-1]

        right = vals[iright[np.argmax(np.cumsum(weights[iright]) > sp.special.erf(sigma / np.sqrt(2)) / 2)]]
        left  = vals[ileft [np.argmax(np.cumsum(weights[ileft])  > sp.special.erf(sigma / np.sqrt(2)) / 2)]]

        return left, mean, right

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
    def get_density_grid(self, param1, param2):
        return gd.plots.MCSampleAnalysis([]).get_density_grid(
            self.get_MCSamples(),
            *[gd.paramnames.ParamInfo(name=p) for p in [param1, param2]],
        )

    def get_contour_vertices(self, sigma, param1, param2):
        density = self.get_density_grid(param1, param2)
        contour_levels = density.getContourLevels([sp.special.erf(sigma / np.sqrt(2))])
        return contourpy.contour_generator(density.x, density.y, density.P).lines(contour_levels)

    def get_peak_2d(self, param1, param2):
        density = self.get_density_grid(param1, param2)
        j, i = np.unravel_index(np.argmax(density.P), density.P.shape)
        return density.x[i], density.y[j]

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

class ImportanceChain(Chain):
    """Description: object to load the importance weights, plot and compute statistics.
    Should be initialized with reference to the respective baseline chain: ImportanceChain(base_chain)
    """

    def __init__(self, filename, base_chain):
        self.filename = filename
        self.name = ".".join(filename.split("/")[-1].split(".")[:-1])
        self.base = base_chain
        self.load_data()

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
        return self.base.get_ranges(*args, **kwargs)

    def get_sampler(self, *args, **kwargs):
        return self.base.get_sampler(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        return self.base.get_params(*args, **kwargs)

    def get_labels(self, *args, **kwargs):
        return self.base.get_labels(*args, **kwargs)

    def get_fiducial(self, *args, **kwargs):
        return self.base.get_fiducial(*args, **kwargs)

    def get_1d_shift(self, param):

        left, right, signal = self.base.get_mean(param), self.get_mean(param), 1
        if left > right:
            left, right = right, left
            signal = -1

        vals = self.base.data[param]
        mask = (vals > left) * (vals < right)

        pval_base = np.sum(self.base.get_weights()[mask]) / np.sum(
            self.base.get_weights()
        )
        pval_is = np.sum(self.get_weights()[mask]) / np.sum(self.get_weights())

        return signal * np.sqrt(2) * sp.special.erfinv(2 * pval_base), signal * np.sqrt(
            2
        ) * sp.special.erfinv(2 * pval_is)

    def get_2d_shift_peak(self, param1, param2, base_posterior=True):
        posterior, peak = self, self.base

        if base_posterior:
            posterior, peak = peak, posterior
            
        return posterior.find_sigma_of_point(
            peak.get_peak_2d(param1, param2), param1, param2
        )

    def get_2d_shift_gaussian(self, param1, param2, base_posterior=True, mean=True):
        inv_cov = np.linalg.inv(
            (self.base if base_posterior else self).get_MCSamples().cov([param1, param2])
        )
        if mean:
            p = self.get_mean([param1, param2]) - self.base.get_mean([param1, param2])
        else:
            p = np.array(self.get_peak_2d(param1, param2)) - \
                np.array(self.base.get_peak_2d(param1, param2))

        return np.einsum("i,ij,j", p, inv_cov, p)

    def get_2d_shift_mean(self, param1, param2, base_posterior=True):
        posterior = self.base if base_posterior else self
        contaminated = self.get_mean([param1, param2])
        baseline = self.base.get_mean([param1, param2])
        return np.abs(
            posterior.find_sigma_of_point(contaminated, param1, param2)
            - posterior.find_sigma_of_point(baseline, param1, param2)
        )

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

    def get_jaccard_index(self, sigma, param1, param2):
        """Returns the Jaccard index between baseline and contaminated n-sigma contours.
        Jaccard index is the ratio intersection/union of the two sets. Its value is 0
        when there is no overlap and 1 when there's complete overlap."""
        import shapely as shp

        poly_baseline = shp.geometry.Polygon(
            self.base.get_contour_vertices(sigma, param1, param2)[0]
        )
        poly_contaminated = shp.geometry.Polygon(
            self.get_contour_vertices(sigma, param1, param2)[0]
        )

        intersection = poly_baseline.intersection(poly_contaminated)
        union = poly_baseline.union(poly_contaminated)
        return intersection.area / union.area

    def load_data(self, *args, **kwargs):
        nsample = self.base.nsample if hasattr(self.base, "nsample") else 0
        super().load_data(*args, **kwargs, nsample=nsample)

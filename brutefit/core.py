import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import itertools as itt
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import permutations, combinations_with_replacement, product
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
from functools import partial

idx = pd.IndexSlice

from .bayesfactor import BayesFactor0
from . import plot 
from .stats import calc_p_zero, weighted_mean, weighted_std, calc_R2, calc_param_pvalues

class Brute():
    """
    Parameters
    ----------
    X : array-like
        An array of covariates (independent variables) of shape (N, M).
    y : array-like
        An array of shape (N,) containing the dependent variable.
    w : array-like
        An array of shape (N,) containing weights to be used in fitting.
        Should be 1 / std**2.
    poly_max : int
        The maximum order of polynomial term to consider.
    max_interaction_order : int
        The highest order of interaction terms to consider.
    permute_interactions : bool
        If True, permutations of interaction terms are tested. Will take longer!
    include_bias : bool
        Whether or not to include a 'bias' (i.e. intercept) term in the fit.
    model : sklearn.linear_model
        An sklearn linear_model, or a custom model object which has a
        .fit(X, y, w) method, and a .score(X, y, w) which returns an unadjusted
        R2 value for the fit. If None, sklearn.linear_model.LinearRegression is used.
    n_processes : int
        Number of multiprocessing threads. Defaults to os.cpu_count()
    chunksize : int
        The size of each subset of jobs passed to multiprocessing threads.
    scale_data : bool
        Whether or not to scale the data before evaluation.
    Scaler : Scaler object
        A class of similar structure to sklearn.preprocessing.StandardScaler.
        An instance must have .fit(), .transform() and .inverse_transform() methods.
    transform : Transformer object
        An object with .transform() and .inverse_transform() methods which will take an
        input array and return a transformed array of the same dimensions.
        If provided, this will be iteratively applied to model variables.
    evaluate_vs_transformed : bool
        If False, all models will be evaluated against the untransformed data. If True,
        models will be evaluated against the transformed data.
    """
    def __init__(self, X, y, w=None, poly_max=1, max_interaction_order=0, permute_interactions=True, 
                 include_bias=True, model=None, n_processes=None, chunksize=None, cmap=None,
                 scale_data=True, Scaler=None, varnames=None, transform=None, fit_vs_transformed=True, evaluate_vs_transformed=False):
        self.X = np.asanyarray(X)
        self.y = np.asanyarray(y)
        if w is None:
            self.w = w
        else:
            self.w = np.asanyarray(w)
        self.poly_max = poly_max
        self.max_interaction_order = max_interaction_order
        self.permute_interactions = permute_interactions
        self.include_bias = include_bias
        self.model = model
        self.n_processes = n_processes
        self.chunksize = chunksize
        
        # check input data
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError('X ({}) and y ({}) must be same length.'.format(self.X.shape[0], self.y.shape[0]))
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

        # perform transformations
        self.transform = transform
        if self.transform is not None:
            if not (hasattr(self.transform, 'transform') and hasattr(self.transform, 'inverse_transform')):
                raise ValueError("transform must have both '.transform()' and '.inverse_transform()' methods.")
            self.tX = self.transform.transform(self.X)
            self.ty = self.transform.transform(self.y)
            self.transformed = True
        else:
            self.transformed = False

        if self.model is None:
            self.model = LinearRegression(fit_intercept=False)

        if self.n_processes is None:
            self.n_processes = cpu_count()

        self.ncov = self.X.shape[-1]
        self.interaction_pairs = np.vstack(np.triu_indices(self.ncov, 1)).T
        self.n_interactions = len(self.interaction_pairs)

        self.make_covariate_names(varnames)
        self.set_covariate_colors(cmap=cmap)

        self.scaled = False
        if scale_data and not self.scaled:
            self.scale_data(Scaler=Scaler)
        
    def make_covariate_names(self, varnames):
        """
        Generate names for model covariates
        """
        self.linear_terms = []
        for o in range(self.poly_max):
            self.linear_terms += ['X{0}^{1}'.format(p, o + 1) for p in range(self.ncov)]

        self.interactive_terms = []
        for o in range(self.max_interaction_order):
            self.interactive_terms += ['X{0}^{2}X{1}^{2}'.format(*ip, o + 1) for ip in self.interaction_pairs]

        self.coef_names = []
        if self.include_bias:
            self.coef_names += ['C']
        self.coef_names += self.linear_terms + self.interactive_terms

        self.xnames = ['X{}'.format(k) for k in range(self.ncov)]

        if varnames is None:
            varnames = self.xnames
        
        self.varnames = varnames
        
        if len(self.varnames) == self.ncov:
            self.vardict = {'X{}'.format(k): v for k, v in enumerate(varnames)}
            real_names = self.coef_names.copy()
            for k, v in self.vardict.items():
                for i, r in enumerate(real_names):
                    real_names[i] = r.replace(k, v)
            for i, r in enumerate(real_names):
                real_names[i] = r.replace('^1', ' ').strip()
            self.vardict = {k: v for k, v in zip(self.coef_names, real_names)}
        else:
            raise ValueError('varnames must be the same length as the number of independent variables ({})'.format(self.ncov))
        if self.include_bias:
            self.vardict['C'] = 'C'
    
    def set_covariate_colors(self, cmap=None):
        n_labels = len(self.varnames)
        if not hasattr(self, 'varcolors'):
            self.varcolors = {}

        if cmap is None:
            # if cmap is not given, set to default color cycler.
            color_cycler = plt.rcParams['axes.prop_cycle']()
            for k in self.vardict.keys():
                self.varcolors[k] = next(color_cycler)['color']
            if 'C' in self.varcolors:
                self.varcolors['C'] = '#666666'
        elif isinstance(cmap, mpl.colors.Colormap):
            for k, i in zip(self.vardict.keys(), np.linspace(0, 1, n_labels)):
                self.varcolors[k] = cmap(i)
            if 'C' in self.varcolors:
                self.varcolors['C'] = '#666666'
        elif isinstance(cmap, dict):
            i_vardict = {v: k for k, v in self.vardict.items()}
            for var, c in cmap.items():
                if var in self.vardict:
                    self.varcolors[var] = c
                elif var in i_vardict:
                    self.varcolors[i_vardict[var]] = c
        else:    
            raise ValueError('`cmap` must be a matplotlib Colormap, an array of colors, or a dict.')

    def scale_data(self, Scaler=None):
        if Scaler is None:
            Scaler = StandardScaler
        self.X_scaler = Scaler().fit(self.X)
        self.X_orig = self.X.copy()
        self.X = self.X_scaler.transform(self.X_orig)

        self.y_scaler = Scaler().fit(self.y)
        self.y_orig = self.y.copy()
        self.y = self.y_scaler.transform(self.y_orig)

        if self.transformed:
            self.tX_scaler = Scaler().fit(self.tX)
            self.tX_orig = self.tX.copy()
            self.tX = self.tX_scaler.transform(self.tX_orig)

            self.ty_scaler = Scaler().fit(self.ty)
            self.ty_orig = self.ty.copy()
            self.ty = self.ty_scaler.transform(self.ty_orig)

        self.scaled = True

    def calc_interaction_permutations(self, c):
        """
        Generates an array of interaction permutations.

        Parameters
        ----------
        c : array-like
            A sequence of N integers specifying the polynomial order
            of each covariate.

        Returns
        -------
        array-like : An array of shape [n_permutations, n_interactions]
        """
        c = np.asanyarray(c)
        max_int_order = min([max(c), self.max_interaction_order])

        n_active = sum(c > 0)
        possible_pairs = np.array(list(itt.combinations(range(len(c)), 2)))

        if n_active < 2:
            interactions = np.zeros((1, possible_pairs.shape[0]), dtype=int)
        else:
            i_active_cov = np.argwhere(c > 0)[:, 0]
            active_pairs = np.array(list(itt.combinations(i_active_cov, 2)))

            i_active_pairs = np.any([np.all(possible_pairs == a, 1) for a in active_pairs], 0)
            n_active_pairs = sum(i_active_pairs)

            interaction_combs = itt.product(range(max_int_order + 1), repeat=n_active_pairs)
            if sum(i_active_pairs) == 1:
                n_interaction_combs = max_int_order + 1
            else:
                n_interaction_combs = (max_int_order + 1)**n_active_pairs

            interactions = np.zeros((n_interaction_combs, i_active_pairs.size), dtype=int)
            interactions[:, i_active_pairs] = list(interaction_combs)

        return interactions

    @staticmethod
    def _comb_long(c, nmax):
        """
        Turn short-form order identifiers into long-form covariate selectors.

        i.e. (0, 1, 2), nmax=2 becomes [False, True, True, False, False, True]
        """
        if nmax == 0:
            return []
        c = np.asanyarray(c)
        return np.concatenate([c >= o + 1 for o in range(nmax)])

    @staticmethod
    def _comb_short(c, ncov):
        """
        Turn long-form covarite selectors into short-form order identifiers.

        i.e. [False, True, True, False, False, True], ncov=3 becomes (0, 1, 2) 
        """
        c = np.asanyarray(c)
        return tuple(c.reshape(len(c) // ncov, ncov).sum(0))

    def calc_model_permutations(self):
        """
        Returns array of (c, interactions) arrays describing all model permutations.
            
        Returns
        -------
        list : Where each item contains (c, interactions).
        """
        combs = itt.product(range(self.poly_max + 1), repeat=self.ncov)

        # calculate all parameter and interaction terms
        pars = []
        for c in combs:
            if self.permute_interactions and self.max_interaction_order > 0:
                interactions = self.calc_interaction_permutations(c)
            else:
                max_int_order = max_int_order = min([max(c), self.max_interaction_order])
                interactions = (np.zeros((max_int_order + 1, self.interaction_pairs.shape[0]), dtype=int) + 
                                np.arange(max_int_order + 1, dtype=int).reshape(-1, 1))
            for i in interactions:
                pars.append(np.concatenate((self._comb_long(c, self.poly_max), self._comb_long(i, self.max_interaction_order))))

        if not self.include_bias:
            pars.remove(pars[0])

        return np.vstack(pars)
    
    def build_desmat(self, X):
        """
        Build design matrix to cover all model permutations
        """
        if self.include_bias:
            desmat = [np.ones(X.shape[0]).reshape(-1, 1)]
        else:
            desmat = []
        
        for o in range(1, self.poly_max + 1):
            desmat.append(X**o)

        for o in range(1, self.max_interaction_order + 1):
            for ip in self.interaction_pairs:
                desmat.append((X[:, ip[0]]**o * X[:, ip[1]]**o).reshape(-1, 1))
        return np.hstack(desmat)

    @staticmethod
    def linear_fit(X, y, w=None, model=None):
        if model is None:
            model = LinearRegression(fit_intercept=False)
        return model.fit(X, y, sample_weight=w)

    @staticmethod
    def _mp_linear_fit(cint, Xd, y, w=None, model=None, include_bias=False, transformer=None, i=0):
        c = cint[i]
        ncov = sum(c == 1)
        if include_bias:
            ind = np.concatenate([[True], c == 1])
        else:
            ind = c == 1
        dX = Xd[:, ind]
        if dX is not None:
            fit = model.fit(dX, y, sample_weight=w)

            coefs = np.full(len(ind), np.nan)
            coefs[ind] = fit.coef_[0]

            pvalues = np.full(len(ind), np.nan)
            pvalues[ind] = calc_param_pvalues(fit.coef_[0], dX, y, fit.predict(dX))

            if transformer is not None:
                # if transformer is provided, back-transform the data before calculating R2
                yp = transformer.inverse_transform(fit.predict(dX))
                ym = transformer.inverse_transform(y)
                R2 = calc_R2(ym, yp)
                # if np.all(c == [True, True, True, False, True, True, False]):
                #     print('trans', R2)

            else:
                R2 = fit.score(dX, y)
                # if np.all(c == np.array([True, True, True, False, True, True, False])):
                #     print('notrans', R2)

            BF = BayesFactor0(dX.shape[0], ncov, R2)

            return i, ncov, R2, BF, coefs, pvalues

    def _fit_polys(self, y, w, permutations, desmat, transformer=None, inds=None, pbar=True, pbar_desc=None):
        total = len(permutations)

        # build partial function for multiprocessing
        pmp_linear_fit = partial(self._mp_linear_fit, permutations, desmat,
                                 y, w, self.model, self.include_bias, transformer)

        # evaluate models
        if self.chunksize is None:
            self.chunksize = min(total // (2 * cpu_count()) + 1, 100)
        # do the work
        if inds is None:
            inds = range(total)
        else:
            total = len(inds)
        with Pool(processes=self.n_processes) as p:
            if pbar:
                pfits = list(tqdm(p.imap(pmp_linear_fit, inds, chunksize=self.chunksize), total=total, desc=pbar_desc, leave=True))
            else:
                pfits = list(p.imap(pmp_linear_fit, inds, chunksize=self.chunksize))
        coefs = [f[-2] for f in pfits]
        pvalues = [f[-1] for f in pfits]
        fits = np.asanyarray([f[:-2] for f in pfits])

        return fits, coefs, pvalues

    def evaluate_polynomials(self, fit_vs_transformed=True, evaluate_vs_transformed=False):
        """
        Evaluate all polynomial combinations and permutations of X against y.
            
        Returns
        -------
        pd.DataFrame : A set of metrics comparing all possible polynomial models.
            Metrics calculated are:
            - R2 : Unadjusted R2 of observed vs. predicted values.
            - BF0 : Bayes factor relative to a null model y = c
            - BF_max : Bayes factor relative to the model with highest BF0.
            BF0 / max(BF0)
            - K : The probability of the best model (M(best)) compared to each 
            other model (M(i)): p(D|M(best)) / p(D|M(i)).
            - evidence : Guidlines for interpreting K, following Kass and Raftery (1995)
        """
        # calculate all parameter and interaction permutations
        self.permutations = self.calc_model_permutations()
        total = len(self.permutations)

        # Build design matrix for most complex model
        self.desmat = self.build_desmat(self.X)                

        # fit all non-transformed models
        fits, coefs, pvalues = self._fit_polys(self.y, self.w, self.permutations, self.desmat, pbar_desc='Evaluating Models:')

        # create output dataframe
        columns = ([('coefs', c) for c in self.coef_names] +
                   [('p_values', c) for c in self.coef_names] +
                   [('metrics', p) for p in ['R2', 'BF0', 'n_covariates']])
        BFs = pd.DataFrame(index=range(total), 
                           columns=pd.MultiIndex.from_tuples(columns))
        
        # assign outputs
        BFs.loc[fits[:, 0].astype(int), [('metrics', 'n_covariates'), ('metrics', 'R2'), ('metrics', 'BF0')]] = fits[:, 1:]
        BFs['coefs'] = np.array(coefs)
        BFs['p_values'] = np.array(pvalues)

        # if transformations are happening...
        self.fit_vs_transformed = fit_vs_transformed
        self.evaluate_vs_transformed = evaluate_vs_transformed
        if self.transformed:
            tcols = [('transformed', k) for k in self.xnames]
            for t in tcols:
                BFs.loc[:, t] = False
            i = BFs.index.max() + 1

            BF_list = [BFs]

            if fit_vs_transformed:
                y = self.ty
            else:
                y = self.y
            
            if evaluate_vs_transformed:
                transformer = None
            else:
                transformer = self.transform
            
            self.trans_desmats = {}

            # Some redundancy here. Could do away with separate calculation of untransformed above, and just evaluate case where
            # all(~tind).
            for tind in tqdm(itt.product([False, True], repeat=self.ncov), total=2**self.ncov, desc='Evaluating Transformed:'):
                atind = np.asanyarray(tind)
                tX = self.X.copy()
                tX[:, atind == 1] = self.tX[:, atind == 1]

                self.trans_desmats[tind] = self.build_desmat(tX)
                # check to see whether transformed variables are active
                # ptind = np.concatenate([atind] * self.poly_max)
                pind = self.permutations[:, :self.ncov][:, atind].sum(1) == sum(atind)
                if any(atind):
                    tfits, tcoefs, tpvalues = self._fit_polys(y, self.w, self.permutations, self.trans_desmats[tind], transformer, np.argwhere(pind)[:,0], pbar=False)
                    n = len(tfits)

                    tBFs = pd.DataFrame(index=range(n), 
                                        columns=pd.MultiIndex.from_tuples(columns + tcols))
                    tBFs.loc[:, [('metrics', 'n_covariates'), ('metrics', 'R2'), ('metrics', 'BF0')]] = tfits[:, 1:]
                    tBFs['coefs'] = np.vstack(tcoefs)
                    tBFs['p_values'] = np.vstack(tpvalues)
                    tBFs.loc[:, idx['transformed', self.xnames]] = atind

                    BF_list.append(tBFs)
                    i += n

            BFs = pd.concat(BF_list, ignore_index=True)

        BFs.loc[:, ('metrics', 'BF_max')] = BFs.loc[:, ('metrics', 'BF0')] / BFs.loc[:, ('metrics', 'BF0')].max() 
        BFs.loc[:, ('metrics', 'K')] = 1 / BFs.loc[:, ('metrics', 'BF_max')]

        BFs.loc[:, ('metrics', 'evidence_against')] = ''
        BFs.loc[BFs.loc[:, ('metrics', 'K')] == 1, ('metrics', 'evidence_against')] = 'Best Model'
        BFs.loc[BFs.loc[:, ('metrics', 'K')] > 1, ('metrics', 'evidence_against')] = 'Not worth more than a bare mention'
        BFs.loc[BFs.loc[:, ('metrics', 'K')] > 3.2, ('metrics', 'evidence_against')] = 'Substantially less probably'
        BFs.loc[BFs.loc[:, ('metrics', 'K')] > 10, ('metrics', 'evidence_against')] = 'Strongly less probably'
        BFs.loc[BFs.loc[:, ('metrics', 'K')] > 100, ('metrics', 'evidence_against')] = 'Decisively less probably'
        
        BFs.sort_values(('metrics', 'K'), inplace=True)
        BFs.reset_index(drop=True, inplace=True)

        self.modelfits = BFs

        return BFs

    def handle_new_data(self, new_data, scaled=False):
        if not scaled and self.scaled:
            new_data = self.X_scaler.transform(new_data)
        
        if self.transformed:
            raise NotImplemented('Cannot do this with transformed data yet...')
        
        return self.build_desmat(new_data)

    def predict(self, new_data=None, scaled=True):
        """
        Calculate predicted y data from all polynomials.

        Parameters
        ----------
        new_data : np.ndarray
            New data used for prediction. If None, we use the
            data provided for fitting.
        """
        if new_data is not None:
            desmat = self.handle_new_data(new_data=new_data, scaled=scaled)
        else:
            desmat = None
        # calculate all predictions
        if self.transformed:
            if desmat is None:
                desmat = self.trans_desmats

            self.pred_all = np.zeros((self.modelfits.shape[0], self.X.shape[0]))
            for t, tdesmat in desmat.items():
                tind = np.all(self.modelfits.transformed == t, 1)
                pred = np.nansum(tdesmat * self.modelfits.coefs.values[tind, np.newaxis, :], axis=2).astype(float)
                if np.any(t):
                    if self.fit_vs_transformed:
                        pred = self.transform.inverse_transform(pred)
                self.pred_all[tind] = pred
        else:
            if desmat is None:
                desmat = self.desmat
            self.pred_all = np.nansum(desmat * self.modelfits.coefs.values[:, np.newaxis, :], axis=2).astype(float)

        # get weights for recombination     
        bf = self.modelfits.metrics.BF_max.values.reshape(-1, 1)

        # un-scale, if appropriate
        if self.scaled:
            self.pred_all_scaled = self.pred_all.copy()
            self.pred_means_scaled = weighted_mean(self.pred_all_scaled, w=bf)
            self.pred_stds_scaled = weighted_std(self.pred_all_scaled, wmean=self.pred_means_scaled, w=bf)

            self.pred_all = self.y_scaler.inverse_transform(self.pred_all_scaled)
        
        self.pred_means = weighted_mean(self.pred_all, w=bf)
        self.pred_stds = weighted_std(self.pred_all, wmean=self.pred_means, w=bf)

    def plot_param_dists(self, xvals=None, bw_method=None, filter_zeros=None, coefs=None, ax=None):
        return plot.parameter_distributions(self, xvals=xvals, bw_method=bw_method, filter_zeros=filter_zeros, coefs=coefs, ax=ax)

    def plot_obs_vs_pred(self, model_ind=None, ax=None, **kwargs):
        return plot.observed_vs_predicted(self, model_ind=model_ind, ax=ax, **kwargs)

    def calc_p_zero(self, bw_method=None):
        return calc_p_zero(self, bw_method)

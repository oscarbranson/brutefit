import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import itertools as itt
from itertools import permutations, combinations_with_replacement, product
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
from functools import partial

from .bayesfactor import BayesFactor0
from . import plot 
from .stats import calc_p_zero, weighted_mean, weighted_std

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
    """
    def __init__(self, X, y, w=None, poly_max=1, max_interaction_order=0, permute_interactions=True, 
                 include_bias=True, model=None, n_processes=None, chunksize=None, 
                 scale_data=True, Scaler=None, varnames=None):
        self.X = np.asanyarray(X)
        self.y = np.asanyarray(y)
        self.w = np.asanyarray(w)
        self.poly_max = poly_max
        self.max_interaction_order = max_interaction_order
        self.permute_interactions = permute_interactions
        self.include_bias = include_bias
        self.model = model
        self.n_processes = n_processes
        self.chunksize = chunksize

        if self.model is None:
            self.model = LinearRegression(fit_intercept=False)

        if self.n_processes is None:
            self.n_processes = cpu_count()

        self.ncov = self.X.shape[-1]
        self.interaction_pairs = np.vstack(np.triu_indices(self.ncov, 1)).T
        
        self.make_covariate_names()

        if varnames is None:
            self.vardict = {'X{}'.format(k): 'X{}'.format(k) for k in range(self.ncov)}
        elif len(varnames) == self.ncov:
            self.vardict = {'X{}'.format(k): v for k, v in enumerate(varnames)}
            real_names = self.coef_names.copy()
            for k, v in self.vardict.items():
                for i, r in enumerate(real_names):
                    real_names[i] = r.replace(k, v)
            for i, r in enumerate(real_names):
                real_names[i] = r.replace('^1', ' ').strip()
            self.vardict.update({k: v for k, v in zip(self.coef_names, real_names)})
        else:
            raise ValueError('varnames must be the same length as the number of independent variables ({})'.format(self.ncov))

        self.scaled = False
        if scale_data and not self.scaled:
            self.scale_data(Scaler=Scaler)
        
    def make_covariate_names(self):
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

    def scale_data(self, Scaler=None):
        if Scaler is None:
            Scaler = StandardScaler
        self.X_scaler = Scaler().fit(self.X)
        self.X_orig = self.X.copy()
        self.X = self.X_scaler.transform(self.X_orig)

        self.y_scaler = Scaler().fit(self.y)
        self.y_orig = self.y.copy()
        self.y = self.y_scaler.transform(self.y_orig)

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
        Turn short-form order strings into long form covariate selectors.

        i.e. (0, 1, 2) becomes [False, True, True, False, False, True]
        """
        if nmax == 0:
            return []
        c = np.asanyarray(c)
        return np.concatenate([c >= o + 1 for o in range(nmax)])

    def calc_model_permutations(self):
        """
        Returns array of (c, interactions) arrays describing all model permutations.
            
        Returns
        -------
        list : Where each item contains (c, interactions) arrays for input
        into build_desmat().
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
                pars.append((self._comb_long(c, self.poly_max), self._comb_long(i, self.max_interaction_order)))

        if not self.include_bias:
            pars.remove(pars[0])

        self.pars = pars
        return pars

    def build_desmat(self, c, interactions=None, include_bias=True):
        """
        Build a design matrix from X for the polynomial specified by c.
        
        The order of the columns returns are:
        [constant, {x_i, ... x_n}**order, {first order interactions}, {second order interactions}]
        
        Parameters
        ----------
        c : array-like
            A sequence of N integers specifying the polynomial order
            of each covariate.
        X : array-like
            An array of covariates of shape (M, N).
        interactions : None or array-like
            If None, no parameter interactions are included.
            If not None, it should be an array of integers the same length as the number
            of combinations of parameters in c, i.e. if c=[1,1,1]: interactions=[1, 1, 1, 1, 1, 1],
            where each integer correspons to the order of the interaction between covariates
            [01, 02, 03, 12, 13, 23].
        include_bias : bool
            Whether or not to inclue a bias (i.e. intercept) in the design matrix.
        """
        c = np.asanyarray(c)
        X = self.X

        interaction_pairs = np.vstack(np.triu_indices(len(c), 1)).T
        if interactions is not None:
            interactions = np.asanyarray(interactions)
            if interaction_pairs.shape[0] != interactions.size:
                msg = '\nIncorrect number of interactions specified. Should be {} for {} covariates.'.format(interaction_pairs.shape[0], c.size)
                msg += '\nSpecifying the orders of interactions between: [' + ', '.join(['{}{}'.format(*i) for i in interaction_pairs]) + ']'
                raise ValueError(msg)
            if interactions.max() > c.max():
                print('WARNING: interactions powers are higher than non-interaction powers.')

        if X.shape[-1] != c.shape[0]:
            raise ValueError('X and c shapes do not not match. X should be (M, N), and c should be (N,).')
        
        if include_bias:
            desmat = [np.ones(X.shape[0]).reshape(-1, 1)]
        else:
            desmat = []

        for o in range(1, c.max() + 1):
            desmat.append(X[:, c>=o]**o)
            
        if interactions is not None:
            for o in range(1, interactions.max() + 1):
                for ip in interaction_pairs[interactions >= o, :]:
                    desmat.append((X[:, ip[0]]**o * X[:, ip[1]]**o).reshape(-1, 1))
        return np.hstack(desmat)
    
    def build_max_desmat(self):
        """
        Build design matrix to cover all model permutations
        """
        if self.include_bias:
            desmat = [np.ones(self.X.shape[0]).reshape(-1, 1)]
        else:
            desmat = []
        
        for o in range(1, self.poly_max + 1):
            desmat.append(self.X**o)

        for o in range(1, self.max_interaction_order + 1):
            for ip in self.interaction_pairs:
                desmat.append((self.X[:, ip[0]]**o * self.X[:, ip[1]]**o).reshape(-1, 1))
        return np.hstack(desmat)

    @staticmethod
    def linear_fit(X, y, w=None, model=None):
        if model is None:
            model = LinearRegression(fit_intercept=False)
        return model.fit(X, y, sample_weight=w)

    @staticmethod
    def _mp_linear_fit(cint, Xd, y, w=None, model=None, include_bias=False, i=0):
        c = np.concatenate(cint[i])
        if include_bias:
            ind = np.concatenate([[True], c == 1])
        else:
            ind = c == 1
        dX = Xd[:, ind]
        if dX is not None:
            ncov = dX.shape[-1] - 1
            fit = model.fit(dX, y, sample_weight=w)
            R2 = fit.score(dX, y, sample_weight=w)
            BF = BayesFactor0(Xd.shape[0], ncov, R2)
            coefs = np.full(len(ind), np.nan)
            coefs[ind] = fit.coef_[0]
            return i, ncov, R2, BF, coefs

    def evaluate_polynomials(self):
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

        # calculate all parameter and interaction terms
        pars = self.calc_model_permutations()
        total = len(pars)

        self.max_desmat = self.build_max_desmat()

        # build partial function for multiprocessing
        pmp_linear_fit = partial(self._mp_linear_fit, pars, self.max_desmat, self.y, self.w, self.model, self.include_bias)

        # evaluate models
        if self.chunksize is None:
            self.chunksize = min(total // (2 * cpu_count()), 100)
        # do the work
        with Pool(processes=self.n_processes) as p:
            fits = list(tqdm(p.imap(pmp_linear_fit, range(total), chunksize=self.chunksize), total=total, desc='Evaluating Models:', leave=False))
            # fits = p.imap(pmp_linear_fit, range(total), chunksize=self.chunksize)
        self.coefs = [f[-1] for f in fits]
        self.fits = np.asanyarray([f[:-1] for f in fits])

        # create output dataframe
        columns = ([('coefs', c) for c in self.coef_names] +
                   [('metrics', p) for p in ['R2', 'BF0', 'n_covariates']])
        BFs = pd.DataFrame(index=range(total), 
                        columns=pd.MultiIndex.from_tuples(columns))
        
        # assign outputs
        BFs.loc[self.fits[:, 0].astype(int), [('metrics', 'n_covariates'), ('metrics', 'R2'), ('metrics', 'BF0')]] = self.fits[:, 1:]
        BFs.loc[:, 'coefs'] = self.coefs

        # BFs.loc[:, ('model', 'include_bias')] = self.include_bias
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

    def predict(self):
        """
        Calculate predicted y data from all polynomials.
        """
        bf = self.modelfits.metrics.BF_max.values.reshape(-1, 1)

        if self.scaled:
            self.pred_all_scaled = np.nansum(self.max_desmat * self.modelfits.coefs.values[:, np.newaxis, :], axis=2).astype(float)
            self.pred_means_scaled = weighted_mean(self.pred_all_scaled, w=bf)
            self.pred_stds_scaled = weighted_std(self.pred_all_scaled, wmean=self.pred_means_scaled, w=bf)

            self.pred_all = self.y_scaler.inverse_transform(self.pred_all_scaled)
            self.pred_means = weighted_mean(self.pred_all, w=bf)
            self.pred_stds = weighted_std(self.pred_all, wmean=self.pred_means, w=bf)
        else:
            self.pred_all = np.nansum(self.max_desmat * self.modelfits.coefs.values[:, np.newaxis, :], axis=2).astype(float)
            self.pred_means = weighted_mean(self.pred_all_scaled, w=bf)
            self.pred_stds = weighted_std(self.pred_all_scaled, wmean=self.pred_means_scaled, w=bf)

    def plot_param_dists(self, xvals=None, bw_method=None, filter_zeros=None, ax=None):
        return plot.parameter_distributions(self, xvals=xvals, bw_method=bw_method, filter_zeros=filter_zeros, ax=ax)

    def plot_obs_vs_pred(self, ax=None):
        return plot.observed_vs_predicted(self, ax=ax)

    def calc_p_zero(self, bw_method=None):
        return calc_p_zero(self, bw_method)

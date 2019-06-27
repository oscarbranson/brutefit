import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from itertools import permutations, combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
from functools import partial

from .bayesfactor import BayesFactor0

def calc_permutations(n, i_max=1):
    """
    Returns all combinations and permutations of n elements with values up to i_max.
    
    Parameters
    ----------
    n : int
        The number of items in the permutations.
    i_max : int
        The maximum value in the polynomial
        
    Returns
    -------
    np.ndarray : An array containing all combinations and permutations of
        covariates orders.
    """
    combs = set()
    for c in combinations_with_replacement(range(i_max+1), n):
        combs.update(permutations(c))
        
    return np.asanyarray(list(combs))

def calc_interaction_permutations(c, interaction_pairs, max_interaction_order=1):
    """
    Generates an array of interaction permutations.

    Parameters
    ----------
    c : array-like
        A sequence of N integers specifying the polynomial order
        of each covariate.
    interaction_pairs : array-like
        An array of valid interaction pairs of shape (n_interactions, 2).
    max_interaction_order : int
        The highest order of interaction terms to consider.

    Returns
    -------
    array-like : An array of shape [n_permutations, n_interactions]
    """

    max_int_order = min([max(c), max_interaction_order])
    n_active = np.sum(c > 0)  # how many covariates are active

    if n_active < 2:
        out = np.zeros((1, interaction_pairs.shape[0]), dtype=int)
    else:
        nc = np.argwhere(c > 0)  # which covariates are active
        # calculate combinations of these covariates
        tmp_int_pairs = np.vstack(np.triu_indices(n_active, 1)).T
        # identify parameter pairs
        int_pairs = np.zeros(tmp_int_pairs.shape, dtype=int)
        for i, co in enumerate(nc[:, 0]):
            int_pairs[tmp_int_pairs == i] = co
        # identify active interactions
        active_ints = np.sum([(interaction_pairs == i).sum(1) == 2 for i in int_pairs], 0, dtype=bool)

        if n_active == 2:
            out = np.zeros((max_int_order + 1, interaction_pairs.shape[0]), dtype=int)
            out[:, active_ints] = np.arange(max_int_order + 1, dtype=int).reshape(-1, 1)
        else:
            # calculate interaction permutations
            int_combs = calc_permutations(np.sum(active_ints), max_int_order)

            out = np.zeros((len(int_combs), interaction_pairs.shape[0]), dtype=int)
            out[:, active_ints] = int_combs

    return out

def calc_model_permutations(ncov, poly_max, max_interaction_order, permute_interactions):
    """
    Returns array of (c, interactions) arrays describing all model permutations.
    
    Parameters
    ----------
    ncov : int
        Number of covariates.
    poly_max : int
        Maximum order of polynomial terms.
    max_interaction_order : int
        Maximum order of interactions terms.
    permute_interactions : bool
        Whether or not to test all permutations of interactive terms.
        
    Returns
    -------
    list : Where each item contains (c, interactions) arrays for input
    into build_desmat().
    """
    combs = calc_permutations(ncov, poly_max)
    interaction_pairs = np.vstack(np.triu_indices(ncov, 1)).T
    
    # calculate all parameter and interaction terms
    pars = []
    for c in tqdm(combs, desc='Calculating Permutations:'):
        if permute_interactions:
            interactions = calc_interaction_permutations(c, interaction_pairs, max_interaction_order)
        else:
            max_int_order = max_int_order = min([max(c), max_interaction_order])
            interactions = (np.zeros((max_int_order + 1, interaction_pairs.shape[0]), dtype=int) + 
                            np.arange(max_int_order + 1, dtype=int).reshape(-1, 1))
        for i in interactions:
            pars.append((c, i))

    return pars


def build_desmat(c, X, interactions=None, include_bias=True):
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
    X = np.asanyarray(X)

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

def linear_fit(X, y, w=None, model=None):
    if model is None:
        model = LinearRegression(fit_intercept=False)
    return model.fit(X, y, sample_weight=w)

def _mp_linear_fit(cint, X, y, w=None, include_bias=True, model=None, i=0):
    dX = build_desmat(c=cint[i][0], X=X, 
                      interactions=cint[i][1], 
                      include_bias=include_bias)
    if dX is not None:
        ncov = dX.shape[-1] - 1
        R2 = model.fit(dX, y, sample_weight=w).score(dX, y, sample_weight=w)
        BF = BayesFactor0(X.shape[0], ncov, R2)
        return i, ncov, R2, BF

def evaluate_polynomials(X, y, w=None, poly_max=1, max_interaction_order=0, permute_interactions=True, 
                         include_bias=True, model=None, n_processes=None, chunksize=None):
    """
    Evaluate all polynomial combinations and permutations of X against y.
    
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

    X = np.asanyarray(X)
    y = np.asanyarray(y)

    if model is None:
        model = LinearRegression(fit_intercept=False)

    # calculate all parameter and interaction terms
    pars = calc_model_permutations(X.shape[-1], poly_max, max_interaction_order, permute_interactions)
    total = len(pars)

    # build partial function for multiprocessing
    pmp_linear_fit = partial(_mp_linear_fit, pars, X, y, w, include_bias, model)

    # evaluate models
    if n_processes is None:
        n_processes = cpu_count()
    if chunksize is None:
        chunksize = min(total // (2 * cpu_count()), 100)
    # do the work
    with Pool(processes=n_processes) as p:
        fits = list(tqdm(p.imap(pmp_linear_fit, range(len(pars)), chunksize=chunksize), total=len(pars), desc='Evaluating Models'))
    fits = np.asanyarray(fits)

    # create output dataframe
    interaction_pairs = np.vstack(np.triu_indices(X.shape[-1], 1)).T
    BFs = pd.DataFrame(index=range(total), 
                       columns=pd.MultiIndex.from_tuples([('orders', 'X{}'.format(p)) for p in range(X.shape[-1])] + 
                                                         [('interactions', 'X{}X{}'.format(*ip)) for ip in interaction_pairs] +
                                                         [('model', p) for p in ['include_bias', 'n_covariates']] +
                                                         [('metrics', p) for p in ['R2', 'BF0']]))
    
    # assign outputs
    BFs.loc[fits[:, 0].astype(int), [('model', 'n_covariates'), ('metrics', 'R2'), ('metrics', 'BF0')]] = fits[:, 1:]
    BFs.loc[fits[:, 0].astype(int), 'orders'] = [p[0] for p in pars]
    BFs.loc[fits[:, 0].astype(int), 'interactions'] = [p[1] for p in pars]

    BFs.loc[:, ('model', 'include_bias')] = include_bias
    BFs.loc[:, ('metrics', 'BF_max')] = BFs.loc[:, ('metrics', 'BF0')] / BFs.loc[:, ('metrics', 'BF0')].max() 
    BFs.loc[:, ('metrics', 'K')] = 1 / BFs.loc[:, ('metrics', 'BF_max')]

    BFs.loc[:, ('metrics', 'evidence_against')] = ''
    BFs.loc[BFs.loc[:, ('metrics', 'K')] == 1, ('metrics', 'evidence_against')] = 'Best Model'
    BFs.loc[BFs.loc[:, ('metrics', 'K')] > 1, ('metrics', 'evidence_against')] = 'Not worth more than a bare mention'
    BFs.loc[BFs.loc[:, ('metrics', 'K')] > 3.2, ('metrics', 'evidence_against')] = 'Substantially less probably'
    BFs.loc[BFs.loc[:, ('metrics', 'K')] > 10, ('metrics', 'evidence_against')] = 'Strongly less probably'
    BFs.loc[BFs.loc[:, ('metrics', 'K')] > 100, ('metrics', 'evidence_against')] = 'Decisively less probably'
    
    return BFs
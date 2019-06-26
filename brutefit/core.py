import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from itertools import permutations, combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .bayesfactor import BayesFactor0

# TODO: Multi-threading

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

def evaluate_polynomials(X, y, w=None, poly_max=1, max_interaction_order=0, permute_interactions=True, include_bias=True, model=None):
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
    mode : sklearn.linear_model
        An sklearn linear_model, or a custom model object which has a
        .fit(X, y, w) method, and a .score(X, y, w) which returns an unadjusted
        R2 value for the fit. If None, sklearn.linear_model.LinearRegression is used.
        
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

    combs = calc_permutations(X.shape[-1], poly_max)
    interaction_pairs = np.vstack(np.triu_indices(X.shape[-1], 1)).T

    # calculate all parameter and interaction terms
    pars = []
    total = 0
    for c in combs:
        if permute_interactions:
            interactions = calc_interaction_permutations(c, interaction_pairs, max_interaction_order)
        else:
            max_int_order = max_int_order = min([max(c), max_interaction_order])
            interactions = (np.zeros((max_int_order + 1, interaction_pairs.shape[0]), dtype=int) + 
                            np.arange(max_int_order + 1, dtype=int).reshape(-1, 1))
        total += interactions.shape[0]
        pars.append((c, interactions))

    BFs = pd.DataFrame(index=range(total), 
                       columns=pd.MultiIndex.from_tuples([('orders', 'X{}'.format(p)) for p in range(X.shape[-1])] + 
                                                         [('interactions', 'X{}X{}'.format(*ip)) for ip in interaction_pairs] +
                                                         [('model', p) for p in ['include_bias', 'n_covariates']] +
                                                         [('metrics', p) for p in ['R2', 'BF0']]))
    i = 0
    pbar = tqdm(total=total)        
    for c, interactions in pars:
        for inter in interactions:
            desmat = build_desmat(c, X, inter, include_bias)

            ncov = desmat.shape[-1] - 1
            BFs.loc[i, ('model', 'n_covariates')] = ncov

            R2 = model.fit(desmat, y, w).score(desmat, y, w)

            BFs.loc[i, 'orders'] = c
            BF0 = BayesFactor0(X.shape[0], ncov, R2)
            BFs.loc[i, ('metrics', 'BF0')] = BF0
            BFs.loc[i, ('metrics', 'R2')] = R2

            BFs.loc[i, 'interactions'] = inter
            i += 1
            pbar.update(1)
    pbar.close()

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

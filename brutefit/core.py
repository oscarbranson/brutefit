import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from itertools import permutations, combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .bayesfactor import BayesFactor0

def calc_permutations(ncov, poly_max=1):
    """
    Returns all combinations and permutations of covariate orders.
    
    Parameters
    ----------
    ncov : int
        The number of covariates in the model.
    poly_max : int
        The maximum order of polynomial term to include.
        
    Returns
    -------
    np.ndarray : An array containing all combinations and permutations of
        covariates orders.
    """
    combs = set()
    for c in combinations_with_replacement(range(poly_max+1), ncov):
        combs.update(permutations(c))
        
    return np.asanyarray(list(combs))


def build_desmat(c, X, interaction_order=0, include_bias=True):
    """
    Build a design matrix from X for the polynomial specified by c.
    
    The contents of the columns of the design matrix varies depending
    on interaction_order. For example, consider X with three columns
    [x, y, z]. For c=[1,2,3] the resulting design matrix could take the 
    following forms:
    interaction_order=0, [1., x, y, z, y**2, z**2, z**3]
    interaction_order=1, [1., x, y, z, x*y, x*z, y*z, y**2, z**2, z**3]
    interaction_order=2, [1., x, y, z, x*y, x*z, y*z, y**2, z**2, y**2 * z**2, z**3]    
    Increasing interaction_order will have no effect unless there is one
    or more covariates with that order or higher - i.e. sum(c >= interaction_order) >= 2
    
    Parameters
    ----------
    c : array-like
        A sequence of N integers specifying the polynomial order
        of each covariate.
    X : array-like
        An array of covariates of shape (M, N).
    interaction_order : int
        The maximum polynomial order of interaction terms to include. 
        Default is zero (no interaction).
    include_bias : bool
        Whether or not to inclue a bias (i.e. intercept) in the design matrix.
    """
    c = np.asanyarray(c)
    X = np.asanyarray(X)
    if X.shape[-1] != c.shape[0]:
        raise ValueError('X and c shapes do not not match. X should be (M, N), and c should be (N,).')
    if include_bias:
        desmat = [np.ones(X.shape[0]).reshape(-1, 1)]
    else:
        desmat = []
    for o in range(1, c.max() + 1):
        if o <= interaction_order:
            desmat.append(PolynomialFeatures(o+1, include_bias=False, interaction_only=True).fit_transform(X[:, c>=o]**o))
        else:
            desmat.append(X[:, c>=o]**o)
    return np.hstack(desmat)

def linear_fit(X, y, w=None, model=None):
    if model is None:
        model = LinearRegression(fit_intercept=False)
    return model.fit(X, y, sample_weight=w)

def evaluate_polynomials(X, y, w=None, poly_max=1, interaction_order=0, include_bias=True, model=None):
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
    interaction_order : int
        The highest order of interaction terms to consider.
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
    BFs = pd.DataFrame(index=range(combs.shape[0]), 
                       columns=pd.MultiIndex.from_tuples([('order', 'X{}'.format(p)) for p in range(X.shape[-1])] + 
                                                         [('model', p) for p in ['interaction_order', 'include_bias', 'n_covariates']] +
                                                         [('metrics', p) for p in ['R2', 'BF0']]))
    
    pbar = tqdm(total=(interaction_order + 1) * combs.shape[0])
    i = 0
    for i_order in range(interaction_order + 1):
        for c in combs:
            BFs.loc[i, 'order'] = c
            desmat = build_desmat(c, X, i_order, include_bias)

            ncov = desmat.shape[-1] - 1
            BFs.loc[i, ('model', 'n_covariates')] = ncov

            R2 = model.fit(desmat, y, w).score(desmat, y, w)

            BF0 = BayesFactor0(X.shape[0], ncov, R2)
            BFs.loc[i, ('metrics', 'BF0')] = BF0
            BFs.loc[i, ('metrics', 'R2')] = R2

            BFs.loc[i, ('model', 'interaction_order')] = i_order
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

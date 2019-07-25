### BROKEN
import numpy as np
import pandas as pd
from sympy import symbols, Add, Mul, S, Equality

def sel_xs(xs, key):
    out = []
    for x, on in zip(xs, key):
        if on:
            out.append(x)
    return out

def sympy_eqn(c, interactions=None, include_bias=True):
    """
    Return a symbolic expression for a polynomial produced by brutefit.
    
    Parameter numbers correspond to column indices in the design matrix.
    
    Parameters
    ----------
    c : array-like
        A sequence of N integers specifying the polynomial order
        of each covariate, OR a row of a dataframe produced by
        evaluate_polynomials.
    interactions : None or array-like
        If None, no parameter interactions are included.
        If not None, it should be an array of integers the same length as the number
        of combinations of parameters in c, i.e. if c=[1,1,1]: interactions=[1, 1, 1, 1, 1, 1],
        where each integer correspons to the order of the interaction between covariates
        [01, 02, 03, 12, 13, 23].
    include_bias : bool
        Whether or not to include a bias term (intercept) in the fit.
    """
    if isinstance(c, pd.core.series.Series):
        interactions = c.loc['interactions'].values
        include_bias = c.loc[('model', 'include_bias')]
        c = c.loc['orders'].values
    c = np.asanyarray(c)

    interaction_pairs = np.vstack(np.triu_indices(len(c), 1)).T
    if interactions is not None:
        interactions = np.asanyarray(interactions)
        if interaction_pairs.shape[0] != interactions.size:
            msg = '\nNot enough interactions specified. Should be {} for {} covariates.'.format(interaction_pairs.shape[0], c.size)
            msg += '\nSpecifying the orders of interactions between: [' + ', '.join(['{}{}'.format(*i) for i in interaction_pairs]) + ']'
            raise ValueError(msg)
        if interactions.max() > c.max():
            print('WARNING: interactions powers are higher than non-interaction powers.')
    
    xs = symbols(['x_{}'.format(i) for i in range(len(c))])

    if include_bias:
        eqn = [symbols('p_0')]
    else:
        eqn = []

    p = 1
    for o in range(1, c.max() + 1):
        for x in sel_xs(xs, c>=o):
            eqn.append(symbols('p_{}'.format(p)) * x**o)
            p += 1

    if interactions is not None:
        for o in range(1, interactions.max() + 1):
            for ip in interaction_pairs[interactions >= o, :]:
                eqn.append(symbols('p_{}'.format(p)) * xs[ip[0]]**o * xs[ip[1]]**o)
                p += 1

    return Equality(symbols('y'), Add(*eqn))
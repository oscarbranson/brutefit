import numpy as np
import pandas as pd
from sympy import symbols, Add, Mul, S, Equality

def sel_xs(xs, key):
    out = []
    for x, on in zip(xs, key):
        if on:
            out.append(x)
    return out

def sympy_eqn(c, interaction_order=0, include_bias=True):
    """
    Return a symbolic expression for a polynomial produced by brutefit.
    
    Parameter numbers correspond to column indices in the design matrix.
    
    Parameters
    ----------
    c : array-like
        A sequence of N integers specifying the polynomial order
        of each covariate, OR a row of a dataframe produced by
        evaluate_polynomials.
    interaction_order : int
        The maximum polynomial order of interaction terms to include. 
        Default is zero (no interaction).
    
    include_bias : bool
        Whether or not to include a bias term (intercept) in the fit.
    """
    if isinstance(c, pd.core.series.Series):
        interaction_order = int(c.model.interaction_order)
        include_bias = c.model.include_bias
        c = c.order.values.astype(int)

    c = np.asanyarray(c)
    xs = symbols(['x_{}'.format(i) for i in range(len(c))])
    terms = []

    p = 0
    if include_bias:
        terms.append(symbols('p_{}'.format(p)))
    #     terms.append(symbols('c'))
        p += 1

    for o in range(1, c.max() + 1):
        for x, on in zip(xs, c >= o):
            if on:
                terms.append(symbols('p_{}'.format(p)) * x**o)
                p += 1
        if o <= interaction_order:
            inds = np.triu_indices(sum(c>=o), 1)
            ixs = sel_xs(xs, c>=o)
            for a, b in zip(*inds):
                terms.append(symbols('p_{}'.format(p)) * ixs[a]**o * ixs[b]**o)
                p += 1

    return Equality(symbols('y'), Add(*terms))
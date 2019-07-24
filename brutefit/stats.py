from scipy import stats
import pandas as pd
import numpy as np

def calc_p_zero(brute, bw_method=None):
    """
    Calculate the probability the contribution of a covariate intersects with zero.

    Warning: this is sensitive to bw_method!
    """
    p_zero = pd.DataFrame(index=brute.coef_names, columns=['p_zero'])

    for c in brute.coef_names:
        
        x = brute.modelfits.loc[:, ('coefs', c)]
        w = brute.modelfits.metrics.BF_max

        w = w[~np.isnan(x)]
        x = x[~np.isnan(x)]

        kde = stats.gaussian_kde(x, bw_method, w)

        p_belowzero = kde.integrate_box_1d(-np.inf, 0)
        p_overzero = kde.integrate_box_1d(0, np.inf)

        p_zero.loc[c, 'p_zero'] = min(p_belowzero, p_overzero)

    return p_zero
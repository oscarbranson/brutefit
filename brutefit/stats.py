from scipy import stats
import pandas as pd
import numpy as np

def weighted_mean(a, w, axis=0):
    return np.sum(a * w, axis=axis) / sum(w)

def weighted_std(a, w, wmean=None, axis=0):
    if wmean is None:
        wmean = weighted_mean(a, w, axis=axis)
    R = (a - wmean).astype(float)
    w = w.astype(float)
    return np.sqrt(np.sum(w * np.power(R, 2), axis=axis) / np.sum(w))

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

def calc_R2(obs, pred):
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

def calc_param_pvalues(params, X, y, ypred):
    """
    Returns t-test p values for all model parameters.
    
    Tests the null hypothesis that the parameter is zero.
    
    Stolen from https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    """
    MSE = (sum((y-ypred)**2))/(X.shape[0] - X.shape[1])

    var_b = MSE*(np.linalg.inv(np.dot(X.T,X)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(X.shape[0]-1))) for i in ts_b]
    
    return p_values
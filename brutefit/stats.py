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

def calc_param_distributions(brute, xvals=None, bw_method=None, filter_zeros=None, coefs=None):
    """
    Calculate weighted density distributions for fitted model parameters.
    """
    model_df = brute.modelfits

    if xvals is None:
        mn = np.nanmin(model_df.coefs)
        mx = np.nanmax(model_df.coefs)
        rn = mx - mn
        pad = 0.05
        xvals = np.linspace(mn - rn * pad, mx + rn * pad, 500)
    
    wts = model_df.loc[:, ('metrics', 'BF_max')].values.astype(float)

    out = pd.DataFrame(index=xvals, columns=model_df.coefs.columns)

    if coefs is None:
        if isinstance(filter_zeros, (int, float)):
            p_zero = calc_p_zero(brute)
            coefs = p_zero.loc[p_zero.p_zero < filter_zeros].index
        else:
            coefs = brute.coef_names
    elif isinstance(coefs, str):
        coefs = [coefs]

    for c in coefs:
        if c in brute.linear_terms:
            line_alpha = 1
            face_alpha = 0.4
            zorder=1
        else:
            zorder=0
            line_alpha = 0.6
            face_alpha = 0.1
        
        cval = model_df.loc[:, ('coefs', c)].values.astype(float)
        ind = ~np.isnan(cval)
        x = cval[ind]
        w = wts[ind]
        # remove values with weights below lower limit to stop kde falling over
        # when sum(w) == 1
        if sum(w) == 1:
            x = x[w > np.finfo(np.float16).tiny]
            w = w[w > np.finfo(np.float16).tiny]

        if len(x) > 1:
            kde = stats.gaussian_kde(x, 
                                     weights=w,
                                     bw_method=bw_method)
            
            # for display purposes only: add some noise to values that are too close together
            # Should never be necessary with noisy data.
            if np.sum(np.diff(x)) < 0.01 * kde.factor:
                x += np.random.normal(0, 0.01 * kde.factor, len(x))
                kde = stats.gaussian_kde(x, weights=w, bw_method=bw_method)

            pdf = kde.evaluate(xvals) * (kde.factor / len(cval))
            out.loc[:, c] = pdf

        elif len(x) == 1:
            # if only one value, draw a sharp distribution
            pdf = stats.norm.pdf(xvals, x, w * (xvals[2] - xvals[0]))
            out.loc[:, c] = pdf
        
    return out

        
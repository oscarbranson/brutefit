from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from .stats import calc_p_zero

def get_limits(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_xlim()
    
    return (min(xlim[0], ylim[0]), max(ylim[1], xlim[1]))

def parameter_distributions(brute, xvals=None, bw_method=None, filter_zeros=None, ax=None):
    """
    Plot a density distribution diagram for fitted models.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    model_df = brute.modelfits

    if xvals is None:
        mn = np.nanmin(model_df.coefs)
        mx = np.nanmax(model_df.coefs)
        rn = mx - mn
        pad = 0.05
        xvals = np.linspace(mn - rn * pad, mx + rn * pad, 500)
    
    wts = model_df.loc[:, ('metrics', 'BF_max')].values.astype(float)

    if isinstance(filter_zeros, (int, float)):
        p_zero = calc_p_zero(brute)
        coefs = p_zero.loc[p_zero.p_zero < filter_zeros].index
    else:
        coefs = brute.coef_names

    for c in coefs:
        if c in brute.linear_terms:
            line_alpha = 1
            face_alpha = 0.4
            zorder=1
        else:
            zorder=0
            line_alpha = 0.6
            face_alpha = 0.1
        
        cval = model_df.loc[:, ('coefs', c)]
        
        if sum(~cval.isnull()) > 1:
                kde = gaussian_kde(cval[~cval.isnull()].values.astype(float), 
                                   weights=wts[~cval.isnull()],
                                   bw_method=bw_method)
                pdf = kde.evaluate(xvals) * (kde.factor / len(cval))

                ax.plot(xvals, pdf, label=brute.vardict[c], alpha=line_alpha, zorder=zorder)
                ax.fill_between(xvals, pdf, alpha=face_alpha, zorder=zorder)
        
    ax.axvline(0, ls='dashed', c=(0,0,0,0.3), zorder=-1)
    ax.set_xlabel('Covariate Influence')
    ax.set_ylabel('Probability Density')

def observed_vs_predicted(brute, ax):
    """
    Plot observed vs. predicted data.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if not hasattr(brute, 'pred_all'):
        brute.predict()

    if brute.scaled:
        ax.errorbar(brute.y_orig, brute.pred_means, xerr=brute.w, yerr=brute.pred_stds, marker='o', lw=0, elinewidth=1)
    else:
        ax.errorbar(brute.y, brute.pred_means, yerr=brute.pred_stds, marker='o', lw=0, elinewidth=1)

    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_aspect(1)

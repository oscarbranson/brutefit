from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
import numpy as np
from .stats import calc_p_zero

def get_limits(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_xlim()
    
    return (min(xlim[0], ylim[0]), max(ylim[1], xlim[1]))

def parameter_distributions(brute, xvals=None, bw_method=None, filter_zeros=None, coefs=None, ax=None):
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
            kde = gaussian_kde(x, 
                               weights=w,
                               bw_method=bw_method)
            
            # for display purposes only: add some noise to values that are too close together
            # Should never be necessary with noisy data.
            if np.sum(np.diff(x)) < 0.01 * kde.factor:
                x += np.random.normal(0, 0.01 * kde.factor, len(x))
                kde = gaussian_kde(x, weights=w, bw_method=bw_method)

            pdf = kde.evaluate(xvals) * (kde.factor / len(cval))
        
        elif len(x) == 1:
            # if only one value, draw a sharp distribution
            pdf = norm.pdf(xvals, x, w * (xvals[2] - xvals[0]))
        else:
            pdf = []
        
        ax.plot(xvals, pdf, label=brute.vardict[c], color=brute.varcolors[c], alpha=line_alpha, zorder=zorder)
        ax.fill_between(xvals, pdf, color=brute.varcolors[c], alpha=face_alpha, zorder=zorder)
    
    # ax.set_ylim(0, ax.get_ylim()[1])
    ax.axvline(0, ls='dashed', c=(0,0,0,0.3), zorder=-1)
    ax.set_xlabel('Covariate Influence')
    ax.set_ylabel('Probability Density')

def observed_vs_predicted(brute, model_ind=None, ax=None, **kwargs):
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
        y = brute.y_orig
    else:
        y = brute.y
    
    if brute.w is not None:
        xerr = (1 / brute.w)**0.5
    else:
        xerr = None

    if model_ind is None:
        ax.scatter(y, brute.pred_means, **kwargs)
        ax.errorbar(y, brute.pred_means, xerr=xerr, yerr=brute.pred_stds, lw=0, elinewidth=1)
    else:
        print(model_ind)
        ax.scatter(y, brute.pred_all[model_ind], **kwargs)
        ax.errorbar(y, brute.pred_all[model_ind], xerr=xerr, lw=0, elinewidth=1)

    ax.set_xlabel('Measured')
    if model_ind is None:
        ax.set_ylabel('Predicted (all models)')
    else:
        ax.set_ylabel(f'Predicted (model {model_ind:.0f})')
    ax.set_aspect(1)

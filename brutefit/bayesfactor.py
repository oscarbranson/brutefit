"""
Code for calculating Bayes Factor given model R2, number of samples and number of covariates.

Method from:
Liang et al. (2008), Mixtures of g Priors for Bayesian Variable Selection. Journal of the American Statistical Association 103, p. 410-423. doi:10.1198/016214507000001337

Also implemented as a web app: http://pcl.missouri.edu/bf-reg

Adapted from R code by David Heslop (david.heslop@anu.edu.au)
"""
import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
import warnings

warnings.simplefilter('ignore')

def rsquared(obs, pred):
    return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

def log1pexp(x):
    if (x<=-37):
        return np.exp(x)
    elif (x<=18):
        return np.log1p(np.exp(x))
    elif (x<=33.3):
        return x+np.exp(-x)
    else:
        return x

def dinvgamma(x,a,b):
    return a * np.log(b) - gammaln(a) - (a+1)*x - b*np.exp(-x)

def integrand(u,N,p,R2,rscaleSqr,log,log_const,shift):
    u += shift
    a = 0.5*((N-p-1) * log1pexp(u) - (N-1)*log1pexp(u + np.log(1-R2)))
    
    shape = 0.5
    scale = rscaleSqr*N/2
    ans = a + dinvgamma(u,shape,scale)-log_const + u
    
    if (log==False):
        return np.exp(ans)
    else:
        return ans
 
def BayesFactor0(N,p,R2,rscale=0.353553390593274):
    """
    Calculated Bayes factor for model.

    Parameters
    ----------
    N : int
        Sample size - the number of observed data.
    p : int
        Number of covariates (i.e. model parameters)
    R2 : float
        The unadjusted R2 of the model
    rscale : float (optional)
        Parameter describing the prior. See paper.

    Returns
    -------
    float : Bayes factor of model
    """
    
    g3 = -(1 - R2) * (p + 3) #* g^3
    g2 = (N - p - 4 - 2 * (1 - R2)) #* g^2
    g1 = (N * (2 - R2) - 3) #*g
    g0 = N

    sol = np.polynomial.polynomial.polyroots([g0, g1, g2, g3])
    modeg = np.real(sol[np.argmin(np.imag(sol)**2)])

    if(modeg<=0):
        modeg = N/20
    
    log_const = integrand(0,N,p,R2,rscale**2,True,0,np.log(modeg))
    h = quad(integrand, -np.inf, np.inf, args=(N,p,R2,rscale**2,False,log_const,np.log(modeg)))[0]
    
    return np.nan_to_num(np.exp(np.log(h) + log_const))
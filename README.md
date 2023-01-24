<div align="right">
  <a href="https://badge.fury.io/py/brutefit"><img src="https://badge.fury.io/py/brutefit.svg" alt="PyPI version" height="18"></a>
  <a href="https://zenodo.org/badge/latestdoi/194186954"><img src="https://zenodo.org/badge/194186954.svg" alt="DOI" height="18"></a>
</div>

# Tired of thinking?

Are you in the business of establishing empirical relationships and then interpolating wildly? Do you struggle to work out which of umpteen different models that describes your data might be 'best'? If so...

## Try BruteFit!<img align="right" width="200" src="img/brute-force.jpg">
BruteFit is an inelegant solution to the age-old question of "Which polynomial best describes my data?" 

If you've got the time and knowledge, you should **definitely** use a [more elegant solution](https://doi.org/10.1111/j.1365-246X.2006.03155.x)... but if not, BruteFit is for you!

BruteFit attempts to fit your data with all combinations and permutations of multivariate polynomials (up to a specified order), with and without permutations of interactive terms (also up to a specified order).

If you have a lot of independent variables, the number of permutations can obviously get out of hand pretty quickly, and this can jam up your computer pretty well for a good while. Beware.

It uses multi-threading to speed things up, but the code is messy and hilariously inneficient... so... well... fix it yourself. Or implement something better.

## Installation
```
pip install brutefit
```

## How it actually works
You give BruteFit:
- Your independent variables as an *(M,N)* array, where *M* is the number of covariates (=independent variables) and *N* is the number of datapoints.
- Your dependent variable as an array with shape *(N,)*.
- Weights used in fitting (![img](http://latex.codecogs.com/svg.latex?%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D)) as an array with shape *(N,)*.
- The maximum order of polynomial terms you'd like to test (`poly_max`).
- The maximum order of interaction terms (`max_interaction_order`).
- Whether or not to test interaction permutations (`permute_interactions`).
- Whether or not to include an intercept term in the fits (`include_bias`).

Brutefit will then loop through *all* permutations of these polynomials, with and without interactive terms.

To evaluate these models it calculates the [Bayes Factor](https://doi.org/10.1080/01621459.1995.10476572) relative to a null model (i.e. *y = c*) using a [this handy little method](https://doi.org/10.1198/016214507000001337). 

## What is this Bayes Factor thing?
The Bayes Factor is a number that tells you *the probability of observing your data if [model X] is true relative to the probability of observing your data if the null model is true.* Or, if you prefer: ![img](http://latex.codecogs.com/gif.latex?B_%7B10%7D%3D%5Cfrac%7Bp%28D%7CM_1%29%7D%7Bp%28D%7CM_0%29%7D). In practical terms, it rewards goodness of fit (i.e. R<sup>2</sup>) and number of data points (*N*), and penalises the model degrees of freedom. So the 'best' model will be that which fits the data well without too many parameters.

Because all these Bayes Factors are calculated relative to the same *null* model, we can then calculate the relative probability of the data given any two other models by ![img](http://latex.codecogs.com/gif.latex?B_%7BNM%7D%3D%5Cfrac%7BB_%7BN0%7D%7D%7BB_%7BM0%7D%7D).

Using this convenient feature, we calculate Bayes Factors for *all* models relative to the 'best' model.

So, what does this number actually *mean*? To ***massively*** over-simplify, your frequentist *p=0.05* [nonsense](https://www.nature.com/news/scientific-method-statistical-errors-1.14700) (or [this](https://www.nature.com/articles/d41586-019-00857-9) or [this](https://www.bmj.com/content/362/bmj.k4039/rr-0) or even [this](https://doi.org/10.1080/00031305.2019.1583913)) would (assuming all assumptions behind the *p* value are valid) correspond to a Bayes Factor of ~20. That is, your alternate hypothesis (*H<sub>1</sub>*) is 20 times more probable than your null hypothesis (*H<sub>0</sub>*). But as I said, this is an enormous and fundamentally invalid comparison... it's just to put the intimidating-sounding Bayes Factor in a possibly more familiar frame of reference.

So *K>20 = ExcellentSignificantPublishInNature* and *K<20 = Weep*? No... The point here is to get away from arbitrary 'significance' cut-offs. But if you *really* want someone else to guide you on this, we can turn to a wonderfully phrased table in [Kass and Raftery (1995)](https://doi.org/10.1080/01621459.1995.10476572), which says:

<table>
<th>K</th><th>Stength of Evidence</th>
<tr>
<td>1 to 3.2</td><td>Not worth mor than a bare mention</td>
</tr>
<tr>
<td>3.2 to 10</td><td>Substantial</td>
</tr>
<tr>
<td>10 to 100</td><td>Strong</td>
</tr>
<tr>
<td>>100</td><td>Decisive</td>
</tr>
</table>


Brutefit does this for you, placing these hugely subjective categories in a handy column for over-interpretation. Note (interestingly) that the criteria for 'decisive' is quite a lot more than a 'significant' p value. Make of that what you will.

## I've run my bazillion models, now what?

At the end of all this, you'll be presented with a wonderful table containing a summary of *all* models. The important columns to glance at are *K* and `evidence_against`, which give the Bayes Factor relative to the 'best' model, and the subjective interpretation of this Bayes Factor. For example, A *K* of *2* for model *M<sub>X</sub>* will mean that the 'best' model is twice as probable as *M<sub>X</sub>*.

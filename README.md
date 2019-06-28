# Tired of thinking?

Are you in the business of establishing empirical relationships and then interpolating wildly? Do you struggle to work out which of umpteen different models that describes your data might be 'best'? If so...

## Try BruteFit!<img align="right" width="200" src="img/brute-force.jpg">
BruteFit is an inelegant solution to the age-old question of "Which polynomial best describes my data?" 

If you've got the time and knowledge, you should **definitely** use a [more elegant solution](https://doi.org/10.1111/j.1365-246X.2006.03155.x)... but if not, BruteFit is for you!

BruteFit attempts to fit your data with all combinations and permutations of multivariate polynomials (up to a specified order), with and without permutations of interactive terms (also up to a specified order).

If you have a lot of independent variables, the number of permutations can obviously get out of hand pretty quickly, and this can jam up your computer pretty well for a good while. Beware.

It uses multi-threading to speed things up, but the code is messy and hilariously inneficient... so... well... fix it yourself. Or implement something better.

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
The Bayes Factor is a number that tells you *the probability of observing your data if [model X] is true relative to the probability of observing your data if the null model is true.* Or, if you prefer: ![img](http://latex.codecogs.com/gif.latex?B_%7B10%7D+%3D+%5Cfrac%7Bp%28D%7CM_1%29%7D%7Bp%28D%7CM_0%29%7D). In practical terms, it rewards goodness of fit (i.e. R<sup>2</sup>) and number of data points (*N*), and penalises the model degrees of freedom. So the 'best' model will be that which fits the data well without too many parameters.

Because all these Bayes Factors are calculated relative to the same *null* model, we can then calculate the relative probability of the data given any two other models by ![img](http://latex.codecogs.com/gif.latex?B_%7BNM%7D+%3D+%5Cfrac%7BB_%7BN0%7D%7D%7BB_%7BM0%7D%7D).

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

# Oh... you want an example?

...

## Fine:

```python
import numpy as np
from brutefit import evaluate_polynomials, sympy_eqn
from brutefit.core import build_desmat

# generate some random data
def calc_combinations(n, k):
    return int(np.math.factorial(n) / np.math.factorial(k) / np.math.factorial(n - k))

ncov = 3  # 3 independent variables
X = np.random.uniform(5,25, (50, ncov))

# now lets generate a random dependent variable
np.random.seed(19)

true_orders = np.random.randint(1, 3, ncov)
true_interactions = np.random.randint(1, 3, calc_combinations(ncov, 2))

# build a design matrix describing our random polynomial
dX = build_desmat(true_orders, X, interactions=true_interactions)

# generate some y data
param_true = np.random.uniform(10, 50, dX.shape[-1])
Y = dX.dot(param_true) + np.random.normal(0, 10, X.shape[0])

# TEST ALL THE MODELS
BFs = evaluate_polynomials(X, Y, poly_max=2, 
                           max_interaction_order=2,
                           permute_interactions=True)
# takes a moment...

# let's look at the top 5 models...
BFs.sort_values(('metrics','K')).head()
```
<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">orders</th>
      <th colspan="3" halign="left">interactions</th>
      <th colspan="2" halign="left">model</th>
      <th colspan="5" halign="left">metrics</th>
    </tr>
    <tr>
      <th></th>
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X0X1</th>
      <th>X0X2</th>
      <th>X1X2</th>
      <th>include_bias</th>
      <th>n_covariates</th>
      <th>R2</th>
      <th>BF0</th>
      <th>BF_max</th>
      <th>K</th>
      <th>evidence_against</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
      <td>1.0</td>
      <td>5.569163e+204</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>Best Model</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>True</td>
      <td>10</td>
      <td>1.0</td>
      <td>1.262466e+199</td>
      <td>2.266886e-06</td>
      <td>4.411337e+05</td>
      <td>Decisively less probably</td>
    </tr>
    <tr>
      <th>170</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>True</td>
      <td>10</td>
      <td>1.0</td>
      <td>1.247264e+199</td>
      <td>2.239590e-06</td>
      <td>4.465102e+05</td>
      <td>Decisively less probably</td>
    </tr>
    <tr>
      <th>224</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>True</td>
      <td>10</td>
      <td>1.0</td>
      <td>1.161021e+199</td>
      <td>2.084731e-06</td>
      <td>4.796783e+05</td>
      <td>Decisively less probably</td>
    </tr>
    <tr>
      <th>233</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>True</td>
      <td>11</td>
      <td>1.0</td>
      <td>3.079288e+193</td>
      <td>5.529176e-12</td>
      <td>1.808588e+11</td>
      <td>Decisively less probably</td>
    </tr>
  </tbody>
</table>

So our 'best' model looks pretty decisive... but what actually *is* the model? The model is described completely in the `orders` and `interactions` sections of this table. Notice that our 'best' model has orders of (2, 1, 2) and interactions of (1, 1, 2)... does this match up with the random values we used to generate our Y data?
```python
# Let's check
# What orders and interactions did we use to generate the Y data?: 
true_orders, true_interactions

   > (array([2, 1, 2]), array([1, 1, 2]))
# YES! It works.
```
So what do these numbers mean? Together, they describe a polynomial:
- `orders` describes the highest degree of polynomial term included for each independent variable. 
- `interactions` describes the highest order of the interaction terms included in the model.

For example, orders = (1, 2, 0) and interactions (1, 1, 0) would give:

*y = p<sub>0</sub> + p<sub>1</sub> X<sub>0</sub> + p<sub>2</sub> X<sub>1</sub> + p<sub>3</sub> X<sub>1</sub><sup>2</sup> + p<sub>4</sub> X<sub>0</sub> X<sub>1</sub> + p<sub>5</sub> X<sub>0</sub> X<sub>2</sub>*

So, what's the equation for our best model? Do we have to work this out manually every time? **NO!!** There's a function for that...
```python
best_model = BFs.loc[BFs.loc[:, ('metrics','K')] == 1]
sympy_eqn(best_model)
```
![img](http://latex.codecogs.com/gif.latex?y+%3D+p_%7B0%7D+%2B+p_%7B1%7D+x_%7B0%7D+%2B+p_%7B2%7D+x_%7B1%7D+%2B+p_%7B3%7D+x_%7B2%7D+%2B+p_%7B4%7D+x_%7B0%7D%5E%7B2%7D+%2B+p_%7B5%7D+x_%7B2%7D%5E%7B2%7D+%2B+p_%7B6%7D+x_%7B0%7D+x_%7B1%7D+%2B+p_%7B7%7D+x_%7B0%7D+x_%7B2%7D+%2B+p_%7B8%7D+x_%7B1%7D+x_%7B2%7D+%2B+p_%7B9%7D+x_%7B1%7D%5E%7B2%7D+x_%7B2%7D%5E%7B2%7D)

Ta daa!
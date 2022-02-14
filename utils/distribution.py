"""
Helpful functions when dealing with distributions
"""

import numpy as np
import time
from .utils import close, closevec
from itertools import combinations
from scipy.optimize import linprog


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)

    return (csprob_n > np_random.rand()).argmax()


def marginal_matrices(n):
    A = np.zeros(shape=(n, n**2))
    B = np.zeros(shape=(n, n**2))
    for i in range(n):
        A[i][i*n:(i+1)*n] = 1
        for j in range(n):
            B[i][j*n+i] = 1
    return A, B


def wass_primal(u, v, d):
    """
    Compute the 1-Wasserstein distance between u (shape=n) and v (shape=n) given the distances matrix d (shape=(n,n)).
    Use the primal formulation.
    """
    n = d.shape[0]
    obj = np.reshape(d, newshape=(n*n))
    A, B = marginal_matrices(n)
    Ae = np.concatenate((A, B), axis=0)
    be = np.concatenate((u, v))
    res = linprog(obj, A_eq=Ae, b_eq=be)
    return res.fun


def wass_dual(u, v, d):
    """
    Compute the 1-Wasserstein distance between u (shape=n) and v (shape=n) given the distances matrix d (shape=(n,n)).
    Use the dual formulation.
    """
    n = d.shape[0]
    comb = np.array(list(combinations(range(n), 2)))
    obj = u - v
    Au = np.zeros(shape=(n*(n-1), n))
    bu = np.zeros(shape=(n*(n-1)))
    for i in range(len(comb)):
        Au[2*i][comb[i][0]] = +1.0
        Au[2*i][comb[i][1]] = -1.0
        Au[2*i+1][comb[i][0]] = -1.0
        Au[2*i+1][comb[i][1]] = +1.0
        bu[2*i] = d[comb[i][0]][comb[i][1]]
        bu[2*i+1] = d[comb[i][0]][comb[i][1]]

    res = linprog(obj, A_ub=Au, b_ub=bu)
    return -res.fun


def random_tabular(size):
    """
    Generate a 1D numpy array whose coefficients sum to 1
    """
    w = np.random.random(size)
    return w / np.sum(w)


def random_constrained(u, d, maxdist):
    """
    Randomly generate a new distribution st the Wasserstein distance between the input
    distribution u and the generated distribution is smaller than the input maxdist.
    The distance is computed w.r.t. the distances matrix d.
    Notice that the generated distribution has the same support as the input distribution.
    """
    max_n_trial = int(1e4) # Maximum number of trials
    val = np.asarray(range(len(u)))
    v = random_tabular(val.size)
    for i in range(max_n_trial):
        if wass_dual(u, v, d) <= maxdist:
            return v
        else:
            v = random_tabular(val.size)
    print('Failed to generate constrained distribution after {} trials'.format(max_n_trial))
    exit()


def clean_distribution(w):
    for i in range(len(w)):
        if close(w[i], 0.0):
            w[i] = 0.0
        else:
            assert w[i] > 0.0, 'Error: negative weight computed ({}th index): w={}'.format(i, w)

    return w


def multinomial_ci(counts, alpha):
    """
    Source https://gist.github.com/chebee7i/9b11dbdd2f4718f9b100

    Calculate a simultaneous (1-alpha) * 100 percent confidence interval.
    Parameters
    ----------
    counts : NumPy array, shape (n,)
        A NumPy array representing counts for each bin. Bins with zero counts
        are allowed and do not affect the confidence intervals for the other
        categories.
    alpha : float
        The alpha value used to construct a (1-alpha) * 100 percent
        simultaneous confidence interval.
    Returns
    -------
    ci : NumPy array, shape (n, 2)
        The confidence intervals for each category. The ith element is the
        confidence interval for the category i. `ci[:, 0]` are the lower
        ends of the confidence intervals while `ci[:, 1]` are the upper ends.
    Examples
    --------
    >>> counts = [10, 12, 15, 5]
    >>> multinomial_ci(counts, .05)
    array([[ 0.08888889,  0.38576742],
           [ 0.13333333,  0.43021186],
           [ 0.2       ,  0.49687853],
           [ 0.04444444,  0.34132297]])
    Adding a category of zero counts does not affect the confidence intervals
    of the other categories, but does yield its own confidence interval.
    >>> counts = [10, 12, 15, 0, 5]
    >>> multinomial_ci(counts, .05)
    array([[ 0.08888889,  0.38576742],
           [ 0.13333333,  0.43021186],
           [ 0.2       ,  0.49687853],
           [ 0.        ,  0.16354519],
           [ 0.04444444,  0.34132297]])
    Notes
    -----
    Until reimplemented in Python, we call R using rpy2.
    """
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector
    multici = importr("MultinomialCI")
    fv = FloatVector(counts)
    ci = multici.multinomialCI(fv, alpha)

    return np.array(ci)

def independent_ci(pmf, N, k, norm=True):
    """
    Source https://gist.github.com/chebee7i/9b11dbdd2f4718f9b100

    Calculate confidence intervals for N independent samples from `pmf`
    Essentially a bootstrapped confidence interval.
    Treat `pmf` as a categorical distribution. Then, take N samples from
    this distribution, yielding a multinomial distribution. For each category i,
    X_i is distributed as a binomial variable with probability of success p_i.
    Thus, the expected number of samples in an infinite number of identical
    experiments, each consisting of N draws, that will be of category i is
    N p_i. Normalizing over the total number of samples N, the probability
    of seeing category i is p_i. Duh. The standard deviation is
    sqrt(N p_i (1-p_i)). Normalizing over the total number of samples N, the
    standard deviation in the probability of seeing category i is
    (p_i (1-p_i)) / sqrt(N).
    This function returns a confidence interval for the number of times
    category i will be seen in N independent trials from `dist` over an
    infinite number of identical experiments, each consisting of N draws.
    The confidence interval is of k standard deviations.  Since counts are
    restricted to be positive and less than or equal to N, the confidence
    interval may be truncated, and hence, asymmetric. The same holds if
    the confidence interval is normalized to yield probabilities.
    Parameters
    ----------
    pmf : NumPy array, shape (n,)
        A NumPy array representing a probability mass function. All elements
        should be within [0,1] and the sum is 1. However, we will normalize
        the array, so passing in counts is also allowed.
    N : int
        The number of independent samples to take.
    k : int
        The number of standard deviations defining the confidence interval. For
        example, for a 95% confidence interval you might set k equal to 1.96.
    norm : bool
        If `True`, then normalize the confidence interval so that it informs
        us about the probability of seeing category i.
    Returns
    -------
    ci : NumPy array, shape (n, 3)
        ci[:, 0] is the expected value for each category.
        ci[:, 1] is the lower end of the confidence interval for each category.
        ci[:, 2] is the upper end of the confidence interval for each category.
    Examples
    --------
    Calculate bootstrapped 95% confidence intervals from sample counts.
    >>> counts = [10, 12, 15, 5]
    >>> independent_ci(counts, sum(counts), 1.96)
    array([[ 0.23809524,  0.109283  ,  0.36690748],
           [ 0.28571429,  0.14908828,  0.4223403 ],
           [ 0.35714286,  0.21222909,  0.50205662],
           [ 0.11904762,  0.02110584,  0.2169894 ]])
    Same as above, but include a category with zero counts. Note that this
    method cannot estimate confidence intervals for bins with zero counts.
    >>> counts = [10, 12, 15, 0, 5]
    >>> independent_ci(counts, sum(counts), 1.96)
    array([[ 0.23809524,  0.109283  ,  0.36690748],
           [ 0.28571429,  0.14908828,  0.4223403 ],
           [ 0.35714286,  0.21222909,  0.50205662],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.11904762,  0.02110584,  0.2169894 ]])
    Now assume the empirical distribution is true and then calculate a 95%
    confidence interval for a series of new experiments, each consisting of
    10000 draws.
    >>> counts = [10, 12, 15, 5]
    >>> independent_ci(counts, 10000, 1.96)
    array([[ 0.22222222,  0.21407372,  0.23037072],
           [ 0.26666667,  0.25799922,  0.27533411],
           [ 0.33333333,  0.3240938 ,  0.34257286],
           [ 0.17777778,  0.1702842 ,  0.18527136]])
    Notes
    -----
    This is not a good method for calculating confidence intervals over
    multinomial distributions. Use `multinomial_ci` instead.
    """
    assert( k > 0 )

    # Normalize just in case counts were passed in.
    pmf = np.array(pmf, copy=True, dtype=float)
    pmf /= pmf.sum()

    # For each category, mean = N * p
    mean = N * pmf
    # For each category, var = N * p * (1-p)
    std = np.sqrt(mean * (1 - pmf))

    ebars = np.array([ mean - k*std, mean + k*std ])
    # Truncate to (0,N)
    ebars[0][ebars[0] < 0] = 0
    ebars[1][ebars[1] > N] = N

    ci = np.array([mean, ebars[0], ebars[1]])
    if norm:
        ci /= N

    return ci.transpose()

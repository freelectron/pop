import numpy as np
# Numerical optimization
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from scipy.stats import entropy

from utils.distribution import *


def project_dichotomy(p_sa_cand, v, p_sa_0, epsilon, lr, D=None,
                      s=None, goal_states=None, dist=wass_dual):
    """
    Dichotomy method of convergence to the correct bounds.
    Move in the direction of the gradient while not exceeding epsilon bounds.
    """
    # If not distance allowed or we dont have gradients
    if close(lr, 0.0, 6):
        return p_sa_cand, lr
    p_sa_sat = np.zeros(len(v))

    ## First worst-case tie-breaking
    p_sa_sat[np.argmin(v)] = 1.0

    ## Closest worst-case tie-breaking
    # assert D[s].shape == v.shape, f"{D[s].shape} != {v.shape}"
    # valid_worst_case_transitions = np.ones((D[s].shape)) * 9999
    # valid_worst_case_transitions[v == v.min()] = D[s][v == v.min()]
    # # Goal state cannot be the worst-case
    # valid_worst_case_transitions[goal_states] = 9999
    # # Find the closest worst-case transition
    # p_sa_sat[np.argmin(valid_worst_case_transitions)] = 1

    # Move mass to the direction of the gradient by linearly combining the masses (affine transformation)
    p_sa_cand_prime = (1 - lr) * p_sa_cand + lr * p_sa_sat
    # Check if you are still within bounds
    cur_dist_val = dist(p_sa_0, p_sa_cand_prime, D)  # dist(p_sa_0, p_sa_cand_prime)  #
    if cur_dist_val <= epsilon:
        p_sa_min = p_sa_cand_prime
    else:
        lr = lr * 0.5
        p_sa_min = p_sa_cand

    # if cur_dist_val <= epsilon:
    #     p_sa_min = p_sa_cand_prime
    # else:
    #     # Go back
    #     p_sa_min = (1 - lr) * p_sa_cand_prime + lr * p_sa_0
    #     lr = lr * 0.5

    return p_sa_min, (close(lr, 0.0, 6) == 0) * lr


def project_dichotomy_same_support(p_sa_cand, v, p_sa_0, epsilon, lr, D=None,
                      s=None, goal_states=None, dist=wass_dual):
    """
    Dichotomy method of convergence to the correct bounds.
    Move in the direction of the gradient while not exceeding epsilon bounds.
    """
    # If not distance allowed or we dont have gradients
    if close(lr, 0.0, 6):
        return p_sa_cand, lr
    p_sa_sat = np.zeros(len(v))

    ## Imortance Sampling: worst-case the from the seen transitions
    v_sup = v.copy()
    v_sup[p_sa_0 > 0] = np.nan
    p_sa_sat[np.nanargmin(v_sup)] = 1

    # Move mass to the direction of the gradient by linearly combining the masses (affine transformation)
    p_sa_cand_prime = (1 - lr) * p_sa_cand + lr * p_sa_sat
    # Check if you are still within bounds
    cur_dist_val = dist(p_sa_0, p_sa_cand_prime, D)  # dist(p_sa_0, p_sa_cand_prime)  #
    if cur_dist_val <= epsilon:
        p_sa_min = p_sa_cand_prime
    else:
        lr = lr * 0.5
        p_sa_min = p_sa_cand

    return p_sa_min, (close(lr, 0.0, 6) == 0) * lr


def wasserstein_worstcase_distribution_analytical(w0, v, c, d, s=None, goal_states=()):  # v, w0, c, d):
    n = len(v)
    if close(c, 0.0) or closevec(v, v[0] * np.ones(n)):
        return w0
    w_worst = np.zeros(n)

    ## First worst-case tie-breaking
    # w_worst[np.argmin(v)] = 1.0

    ## Random tie-breaking
    # lowest_vals_mask = v == v.min()
    # w_worst[lowest_vals_mask] = 1 / len(lowest_vals_mask)

    ## Closest worst-case tie-breaking
    assert d[s].shape == v.shape
    valid_worst_case_transitions = np.ones((d[s].shape)) * 9999
    valid_worst_case_transitions[v == v.min()] = d[s][v == v.min()]
    # Goal state cannot be the worst-case
    valid_worst_case_transitions[goal_states] = 9999
    # Find the closest worst-case transition
    w_worst[np.argmin(valid_worst_case_transitions)] = 1

    # DEBUGGING/TESTING
    # with np.printoptions(precision=2, suppress=False):
        # print(f'We are in the state {s} doing the projection.')
        # print(f'The valid worst-case transitions vector \n {valid_worst_case_transitions.reshape((5,7))}.')
        # print(f'Nambla_p vector \n {w_worst.reshape((5,7))}.')
    distance = wass_dual(w_worst, w0, d)
    if distance <= c:

        # # DEBUGGING/TESTING
        # with np.printoptions(precision=2, suppress=False):
        #     print(f'p_0_sa \n {w0.reshape((5, 7))}. ')
        #     print(f'p_min_sa \n {w_worst.reshape((5, 7))}. ')

        return w_worst

    lbd = c / distance
    w = w_an = (1.0 - lbd) * w0 + lbd * w_worst

    ##DEBUGGING/TESTING
    # with np.printoptions(precision=2, suppress=False):
    #     print(f'p_0_sa \n {w0.reshape((5, 7))}.')
    #     print(f'p_min_sa \n {w.reshape((5, 7))}.')

    return clean_distribution(w)


def wasserstein_worstcase_distribution_bisection(p_0_sa, v, epsilon, D):
    """
    Bisection method described from NSMDP paper.
    Note: not yet tetsed.
    """
    time_start = time.time()
    n = len(v)
    if n > 28:
        print('WARNING: solver instabilities above this number of dimensions (n={})'.format(n))
    if close(epsilon, 0.0) or closevec(v, v[0] * np.ones(n)):
        return p_0_sa
    w_worst = np.zeros(n)
    w_worst[np.argmin(v)] = 1.0
    if (wass_dual(w_worst, p_0_sa, D) <= epsilon):
        return w_worst
    else:
        wmax = w_worst
        wmin = p_0_sa
        w = 0.5 * (wmin + wmax)
        for i in range(1000):  # max iter is 1000
            if (wass_dual(w, p_0_sa, D) <= epsilon):
                wmin = w
                wnew = 0.5 * (wmin + wmax)
            else:
                wmax = w
                wnew = 0.5 * (wmin + wmax)
            if closevec(wnew, w, 6):  # precision is 1e-6
                w = wnew
                break
            else:
                w = wnew

    return clean_distribution(w)


def worstcase_distribution_linear_program(p_0_sa, v, epsilon_robust):
    """
    Solving with linear programming (LP).

    - Vector `v` can be replaced with r_sa.
    - Very slow if running on the grid of shape 5 x 5.
    - Needs clean(res.x) function that makes very small numbers zeros.
    Args:
        p_0_sa (numpy.array): discrete probability mass function (vector of probs) aka reference trans. prob.
        v (numpy.array): vector of state values
        epsilon_robust (float): epsilon error bound value
    Returns:
        numpy.array, discrete probability that minimizes the state-action value function for that s,a
    """
    calc_expected_gain = lambda p_sa: p_sa.T.dot(v)

    def _calc_distance_l1(p_sa):
        """
        Note: p_0_sa is defined outside of the function.
        """
        return -cdist(p_sa.reshape((1, -1)), p_0_sa.reshape((1, -1)), 'cityblock')[0] + epsilon_robust

    # Constraints
    distance_constraint = {'type': 'ineq', 'fun': _calc_distance_l1}
    # We nu_s_a is a pmf, so entries should sum to one
    calc_pmf_sum = lambda p_sa: p_sa.sum() - 1
    pmf_constraint = {'type': 'eq', 'fun': calc_pmf_sum}
    # All entries nonnegative
    bounds = tuple([(0, 1) for _ in p_0_sa])
    res = minimize(calc_expected_gain, p_0_sa, method='trust-constr',
                   constraints=[pmf_constraint, distance_constraint],
                   options={'verbose': 0, 'disp': False},
                   bounds=bounds)

    return res.x


def l1_worstcase_distribution(p_0_sa, v, epsilon_robust):
    """
    Solve for L1 loss specifially.
    You shift all the mass to the state with the worst-possible transition value. This is equivalent to finding the
    worst-case transition within L1 distance.

    - Vector `v` can be replaced with r_sa.
    Args:
        p_0_sa (numpy.array): discrete probability mass function (vector of probs) aka reference trans. prob.
        v (numpy.array): vector of state values
        epsilon_robust (float): epsilon error bound value
    Returns:
        numpy.array, discrete probability that minimizes the state-action value function for that s,a
    """

    # Calculate difference
    def _calc_distance_l1(p_sa_0, p_sa_1):
        return cdist(p_sa_0.reshape((1, -1)), p_sa_1.reshape((1, -1)), 'cityblock')[0]

    # Prob. distribution with the lowest performance
    argmin_v = np.argmin(v)
    # Vector where the lowest return is acheived
    p_sat = np.identity(len(v))[argmin_v]

    if _calc_distance_l1(p_sat, p_0_sa) <= epsilon_robust:
        p_sa_min = p_sat
    else:
        # Shift all mass to the point where the lowest return is achieved (lowest v)
        p_sa_min = np.copy(p_0_sa)
        # How much we are allowed to change the distribution
        delta_change = epsilon_robust
        while True:
            p_sa_min_old = np.copy(p_sa_min)
            # Highest return possible transition
            argmax_v = np.argmax(np.where(p_sa_min_old > 0, v, -np.inf))
            delta = delta_change / 2 if p_sa_min_old[argmax_v] > delta_change / 2 else p_sa_min_old[argmax_v]
            p_sa_min[argmax_v] -= delta
            p_sa_min[argmin_v] += delta
            delta_change = delta_change - 2.0 * delta
            # If moved all the mass we could - break
            if delta_change == 0:
                break

    return p_sa_min

import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty.")
    if not (0 < theta < 1):
        raise ValueError("Theta must be in the interval (0, 1).")
    
    data_array = np.array(data)
    if not np.all(np.isin(data_array, [0, 1])):
        raise ValueError("Data must only contain 0s and 1s.")

    # sum(x_i * log(theta) + (1 - x_i) * log(1 - theta))
    successes = np.sum(data_array)
    failures = len(data_array) - successes
    return successes * math.log(theta) + failures * math.log(1 - theta)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty.")
    
    data_array = np.array(data)
    if not np.all(np.isin(data_array, [0, 1])):
        raise ValueError("Data must only contain 0s and 1s.")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    num_successes = int(np.sum(data_array))
    num_failures = len(data_array) - num_successes
    mle = num_successes / len(data_array)

    ll_map = {}
    best_candidate = None
    max_ll = -float('inf')

    for t in candidate_thetas:
        ll = bernoulli_log_likelihood(data, t)
        ll_map[t] = ll
        if ll > max_ll:
            max_ll = ll
            best_candidate = t

    return {
        'mle': mle,
        'num_successes': num_successes,
        'num_failures': num_failures,
        'log_likelihoods': ll_map,
        'best_candidate': best_candidate
    }


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty.")
    if lam <= 0:
        raise ValueError("Lambda must be greater than 0.")
    
    data_array = np.array(data)
    if np.any(data_array < 0) or np.any(data_array % 1 != 0):
        raise ValueError("Data must contain nonnegative integers.")

    # sum(x_i * log(lam) - lam - log(x_i!))
    # Using lgamma(x + 1) for log(x!)
    log_likelihood = 0.0
    for x in data_array:
        log_likelihood += (x * math.log(lam)) - lam - math.lgamma(x + 1)
    
    return log_likelihood


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty.")
    
    data_array = np.array(data)
    if np.any(data_array < 0) or np.any(data_array % 1 != 0):
        raise ValueError("Data must contain nonnegative integers.")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    n = len(data_array)
    total_count = int(np.sum(data_array))
    sample_mean = total_count / n
    mle = sample_mean

    ll_map = {}
    best_candidate = None
    max_ll = -float('inf')

    for l in candidate_lambdas:
        ll = poisson_log_likelihood(data, l)
        ll_map[l] = ll
        if ll > max_ll:
            max_ll = ll
            best_candidate = l

    return {
        'mle': mle,
        'sample_mean': sample_mean,
        'total_count': total_count,
        'n': n,
        'log_likelihoods': ll_map,
        'best_candidate': best_candidate
    }

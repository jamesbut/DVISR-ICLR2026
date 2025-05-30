import math
import numpy as np
import scipy


# Posterior parameter solutions for y = c and a prior over c
def post_params_analytic_c(prior_mean, prior_sd, likelihood_sd, N, x_bar):

    post_var = 1 / ((N / likelihood_sd ** 2) + (1 / prior_sd ** 2))
    post_sd = math.sqrt(post_var)

    post_mean = post_var * ((prior_mean / prior_sd ** 2)
                            + (N * x_bar / likelihood_sd ** 2))

    return post_mean, post_sd


# Posterior parameter solutions for y = cx and a prior over c
def post_params_analytic_cx(prior_mean, prior_sd, likelihood_sd, data):

    sum_x_squared = sum(x ** 2 for x in data['x'])
    sum_x_y = sum(x * y for x, y in zip(data['x'], data['y']))

    post_var = 1 / ((1 / prior_sd ** 2)
                    + (1 / likelihood_sd ** 2) * sum_x_squared)
    post_sd = math.sqrt(post_var)

    post_mean = post_var * ((prior_mean / prior_sd ** 2)
                            + (1 / likelihood_sd ** 2) * sum_x_y)

    return post_mean, post_sd


# Compute evidence analytically for the same situation as above using
# the parameters of the posterior calculated analytically
def analytic_evidence_post_params(post_mean, post_sd, all_exprs, alg,
                                  likelihood, log_likelihood, data):

    post_gauss = scipy.stats.norm(post_mean, post_sd)

    z = all_exprs[0]

    ev = (likelihood(data, z, alg._likelihood_sd,
                     alg._max_num_tokens, alg._net_masks)
          * alg._prior(z) / post_gauss.pdf(z.tokens()[0]['value']))

    return ev


def analytic_log_evidence_post_params(post_mean, post_sd, all_exprs, alg,
                                      likelihood, log_likelihood, data):
    return np.log(analytic_evidence_post_params(
        post_mean, post_sd, all_exprs, alg, likelihood, log_likelihood, data
    ))


# Attempt to calculate the evidence analytically if the expressions fit
# a certain format
def analytic_evidence(exprs, alg):

    '''
    def joint(e, alg):



    p_x = []

    for e in expr:

        # Check whether evidence can be calculated analytically for each
        # expression

        # If there are no distributional constants, evidence is simple
        if e.num_distr_consts() == 0:
            p_x.append()
    '''


# Compute evidence analytically for the same situation as above.
# This code was given by ChatGPT and I suspect it is incorrect.
# NOTE: I just think this isn't correct.
def analytic_evidence_incorrect(x, sigma2, mu0, sigma0_2):
    """
    Computes the marginal likelihood (evidence) p(x)
    for Gaussian likelihood with known variance and Gaussian prior on the mean.

    Parameters:
    - x : array_like, shape (n,)
        Observed data
    - sigma2 : float
        Known variance of the likelihood
    - mu0 : float
        Prior mean
    - sigma0_2 : float
        Prior variance

    Returns:
    - evidence : float
        The value of the marginal likelihood p(x)
    """
    x = np.asarray(x)
    n = len(x)
    x_bar = np.mean(x)

    # Compute parts
    data_term = np.sum((x - x_bar) ** 2)
    norm_likelihood = (1 / np.sqrt(2 * np.pi * sigma2)) ** n
    exp_likelihood = np.exp(-0.5 * data_term / sigma2)

    combined_var = sigma0_2 + sigma2 / n
    norm_prior = 1 / np.sqrt(2 * np.pi * combined_var)
    exp_prior = np.exp(-0.5 * (x_bar - mu0) ** 2 / combined_var)

    # Evidence is product of likelihood and prior integrals
    evidence = norm_likelihood * exp_likelihood * norm_prior * exp_prior
    return evidence


def analytic_log_evidence_incorrect(x, sigma2, mu0, sigma0_2):
    return np.log(analytic_evidence_incorrect(x, sigma2, mu0, sigma0_2))

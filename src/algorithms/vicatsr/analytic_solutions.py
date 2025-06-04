import numpy as np
import scipy


# Posterior parameter solutions for y = c and a prior over c
def post_params_analytic_c(prior_mean, prior_sd, likelihood_sd, N, x_bar):

    post_var = 1 / ((N / likelihood_sd ** 2) + (1 / prior_sd ** 2))
    post_sd = np.sqrt(post_var)

    post_mean = post_var * ((prior_mean / prior_sd ** 2)
                            + (N * x_bar / likelihood_sd ** 2))

    return post_mean, post_sd


# Posterior parameter solutions for y = cx and a prior over c
def post_params_analytic_cx(prior_mean, prior_sd, likelihood_sd, data):

    sum_x_squared = sum(x ** 2 for x in data['x'])[0]
    sum_x_y = sum(x * y for x, y in zip(data['x'], data['y']))[0]

    post_var = 1 / ((1 / prior_sd ** 2)
                    + (1 / likelihood_sd ** 2) * sum_x_squared)
    post_sd = np.sqrt(post_var)

    post_mean = post_var * ((prior_mean / prior_sd ** 2)
                            + (1 / likelihood_sd ** 2) * sum_x_y)

    return post_mean, post_sd


# Posterior parameter solutions for y = c + x and a prior over c
def post_params_analytic_c_plus_x(prior_mean, prior_sd, likelihood_sd, data):

    N = len(data['y'])

    sum_diffs = sum(y - x for x, y in zip(data['x'], data['y']))[0]

    post_var = 1 / ((N / likelihood_sd ** 2) + (1 / prior_sd ** 2))
    post_sd = np.sqrt(post_var)

    post_mean = post_var * ((prior_mean / prior_sd ** 2)
                            + (1 / likelihood_sd ** 2) * sum_diffs)

    return post_mean, post_sd


# Compute evidence analytically when the posterior parameters have
# been obtained
def analytic_evidence_post_params(post_mean, post_sd, z, vicatsr):

    post_gauss = scipy.stats.norm(post_mean, post_sd)

    # Find float const token value
    value = next((t['value'] for t in z.tokens()
                  if t['sub_type'] == 'float_const'), None)

    # Calculate evidence by dividing joint by posterior
    ev = vicatsr.joint(z, vicatsr._data) / post_gauss.pdf(value)

    return ev


def analytic_log_evidence_post_params(post_mean, post_sd, z, vicatsr):
    return np.log(analytic_evidence_post_params(
        post_mean, post_sd, z, vicatsr
    ))


# Attempt to calculate the evidence analytically if the expressions are of a
# certain form.
# Return None if analytic evidence could not be calculated.
def analytic_evidence(exprs, vicatsr):

    p_x = []

    # Check whether evidence can be calculated analytically for each
    # expression and calculate it
    for z in exprs:

        # If there are no distributional constants, evidence contribution
        # is simply the joint
        if z.num_distr_consts() == 0:
            p_x.append(vicatsr.joint(z, vicatsr._data))

        # Otherwise determine the form of the expression and then calculate
        # evidence contribution accordingly
        else:

            # If expression is y = c
            if (z.num_tokens() == 1
                and z.tokens()[0]['sub_type'] == 'float_const'):

                N = len(vicatsr._data['y'])
                y_bar = np.mean(vicatsr._data['y'])

                post_params = post_params_analytic_c(
                    vicatsr._prior_mean, vicatsr._prior_sd,
                    vicatsr._likelihood_sd, N, y_bar
                )

                p_x.append(
                    analytic_evidence_post_params(
                        post_params[0], post_params[1], z, vicatsr
                    )
                )

            # If expression is y = cx
            elif (z.num_tokens() == 3
                  and any(t['op'] == '*' for t in z.tokens())
                  and any(t['sub_type'] == 'float_const' for t in z.tokens())
                  and any(t['sub_type'] == 'var_const' for t in z.tokens())):

                post_params = post_params_analytic_cx(
                    vicatsr._prior_mean,
                    vicatsr._prior_sd,
                    vicatsr._likelihood_sd,
                    vicatsr._data)

                p_x.append(
                    analytic_evidence_post_params(
                        post_params[0], post_params[1], z, vicatsr
                    )
                )

            # If expression is y = c + x
            elif (z.num_tokens() == 3
                  and any(t['op'] == '+' for t in z.tokens())
                  and any(t['sub_type'] == 'float_const' for t in z.tokens())
                  and any(t['sub_type'] == 'var_const' for t in z.tokens())):

                post_params = post_params_analytic_c_plus_x(
                    vicatsr._prior_mean,
                    vicatsr._prior_sd,
                    vicatsr._likelihood_sd,
                    vicatsr._data)

                p_x.append(
                    analytic_evidence_post_params(
                        post_params[0], post_params[1], z, vicatsr
                    )
                )

            else:

                # Check whether z is a invalid expression under the constraints.
                # If invalid, it will not contribute anything to the evidence.
                # Do not return None.
                if not z.valid_eq(vicatsr._max_num_tokens, vicatsr._net_masks):
                    continue

                # If it is valid but just cannot be calculated then return None
                else:
                    return None

    return sum(p_x)


def analytic_log_evidence(exprs, vicatsr):
    ev = analytic_evidence(exprs, vicatsr)
    return np.log(ev) if ev else None


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

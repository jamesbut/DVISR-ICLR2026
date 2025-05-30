import copy
import numpy as np
import scipy
import itertools


# Integrate q(z,c) w.r.t all c values for a set of expressions, exprs.
# Returns list of q(z) values for each expression.
def integrate_q_z_c(q, exprs):

    def qz_func(*args):

        # Unpack arguments
        z = args[-1]
        cs = args[:-1]

        # Set consts
        z = copy.copy(z)
        z.set_distr_consts(cs)

        return q.pdf(z)

    q_zs = []
    for i, z in enumerate(exprs):

        # Create integration bounds for continuous parameters
        integration_bounds = [[-np.inf, np.inf]
                              for _ in range(z.num_distr_consts())]

        # If z has no distributional constants, no need to integrate
        if z.num_distr_consts() == 0:
            q_zs.append(q.pdf(z).item())

        else:
            res, error = scipy.integrate.nquad(qz_func,
                                               integration_bounds,
                                               args=(z,))
            q_zs.append(res)

    return q_zs


# Integrate p(z,c|x) w.r.t all c values for a set of expressions.
# Returns list of p(z|x) values for each expression.
def integrate_p_z_c_x(vicatsr, exprs):

    def post_func(*args):

        # Unpack arguments
        z = args[-1]
        cs = args[:-1]

        # Set consts
        z = copy.copy(z)
        z.set_distr_consts(cs)

        return vicatsr.posterior(
            vicatsr._data, z, vicatsr._enumerate_expressions(vicatsr._data)
        )

    p_z_x = []
    for i, z in enumerate(exprs):

        # Create integration bounds for continuous parameters
        integration_bounds = [[-np.inf, np.inf]
                              for _ in range(z.num_distr_consts())]

        # If z has no distributional constants, no need to integrate
        if z.num_distr_consts() == 0:
            p_z_x.append(vicatsr.posterior(vicatsr._data, z, exprs))

        else:
            res, error = scipy.integrate.nquad(post_func,
                                               integration_bounds,
                                               args=(z,))
            p_z_x.append(res)

    return p_z_x


# Integrate the joint, p(z,c,x), w.r.t z and c.
# This results in p(x), the evidence.
def integrate_joint(vicatsr, data, zs, int_method, log_space, int_error_tol):

    num_distr_consts = [e.num_distr_consts() for e in zs]
    total_num_distr_consts = sum(num_distr_consts)

    def joint_func_no_split(*args):

        # Unpack arguments
        num_consts = args[-1]
        cumm_num_consts = [0] + list(itertools.accumulate(num_consts))
        total_num_consts = sum(num_consts)
        x = args[:total_num_consts + 1]
        all_exps = args[total_num_consts + 1]

        # Sample a particular expression
        samp = x[0]
        idx = int(samp)

        # This might happen if the integrator samples exactly the
        # upper bound
        if idx >= len(all_exps):
            return 0.0

        z = copy.copy(all_exps[idx])

        # Parse consts relevant to selected expression
        this_z_consts = x[cumm_num_consts[idx] + 1:
                          cumm_num_consts[idx + 1] + 1]
        other_z_consts = x[1:cumm_num_consts[idx] + 1] \
                         + x[cumm_num_consts[idx + 1] + 1:]

        if z.num_distr_consts() > 0:
            z.set_distr_consts(this_z_consts)

        if any(c < 0.0 or c > 1.0 for c in other_z_consts):
            return 0.0

        if log_space:
            return vicatsr.joint_log_space(z, data)
        else:
            return vicatsr.joint(z, data)

    def joint_func_all_c(*args):

        # Unpack arguments
        num_consts = args[-1]
        idx = args[-2]
        z = args[-3]
        cumm_num_consts = [0] + list(itertools.accumulate(num_consts))
        x = args[:-3]

        z = copy.copy(z)

        # Parse consts relevant to selected expression
        this_z_consts = x[cumm_num_consts[idx]:
                          cumm_num_consts[idx + 1]]
        other_z_consts = x[:cumm_num_consts[idx]] \
                         + x[cumm_num_consts[idx + 1]:]

        if z.num_distr_consts() > 0:
            z.set_distr_consts(this_z_consts)

        if any(c < 0.0 or c > 1.0 for c in other_z_consts):
            return 0.0

        if log_space:
            return vicatsr.joint_log_space(z, data)
        else:
            return vicatsr.joint(z, data)

    def joint_func(*args):

        # Unpack arguments
        z = args[-1]
        cs = args[:-1]

        # Set consts
        z = copy.copy(z)
        z.set_distr_consts(cs)

        if log_space:
            return vicatsr.joint_log_space(z, data)
        else:
            return vicatsr.joint(z, data)

    # Sum over expressions is separated from the integration
    if int_method == 'split_sum':

        # Create integration bounds for continuous parameters
        integration_bounds = [[-np.inf, np.inf]
                              for _ in range(total_num_distr_consts)]

        p_x = 0.0
        for i, z in enumerate(zs):

            z = copy.deepcopy(z)

            # If z has no distributional constants, no need to integrate
            if z.num_distr_consts() == 0:
                if log_space:
                    p_x += vicatsr.joint_log_space(z, data)
                else:
                    p_x += vicatsr.joint(z, data)

            else:
                res, error = scipy.integrate.nquad(joint_func_all_c,
                                                   integration_bounds,
                                                   args=(z, i,
                                                         num_distr_consts))
                p_x += res

    # Only integrate over the c values in the particular z in question
    elif int_method == 'only_own_c':

        p_x = 0.0
        for i, z in enumerate(zs):

            z = copy.deepcopy(z)

            # Create integration bounds for continuous parameters
            integration_bounds = [[-np.inf, np.inf]
                                  for _ in range(z.num_distr_consts())]

            # Reduce tolerance for error within integrator.
            # When the evidence, p(x), is very small the integrator
            # produced a large relative error, resulting in inaccurate
            # evidence values.
            # The default values of the below quantities are 1.49e-8.
            if int_error_tol:
                opts = [{'epsabs': int_error_tol, 'epsrel': int_error_tol}
                        for _ in range(z.num_distr_consts())]
            else:
                opts = None

            # If z has no distributional constants, no need to integrate
            if z.num_distr_consts() == 0:
                p_x += vicatsr.joint_log_space(z, data)

            else:
                res, error = scipy.integrate.nquad(joint_func,
                                                   integration_bounds,
                                                   args=(z,),
                                                   opts=opts)
                p_x += res

    else:

        # Create integration bounds
        # The first bound is for selecting the particular expression
        # The remaining bounds are for each of the optimisable constants
        integration_bounds = [[0, len(zs)]]

        for i in range(total_num_distr_consts):
            integration_bounds.append([-np.inf, np.inf])

        p_x, error = scipy.integrate.nquad(joint_func_no_split,
                                           integration_bounds,
                                           args=(zs, num_distr_consts))

    return p_x


# Integrate posterior w.r.t z, c and x.
# This should integrate to 1.0 if everything is working as expected
def integrate_posterior(vicatsr, exprs, ev=None):

    def posterior_func(*args):

        c = args[:-4]
        exprs = args[-3]
        ev = args[-2]
        vicatsr = args[-1]

        z = copy.copy(args[-4])
        z.set_distr_consts(c)

        return vicatsr.posterior_log_space(vicatsr._data, z, exprs, ev)

    int_p_z_x = 0.0
    for z in exprs:

        # Create integration bounds for continuous parameters
        integration_bounds = [[-np.inf, np.inf]
                              for _ in range(z.num_distr_consts())]

        # If z has no distributional constants, no need to integrate
        if z.num_distr_consts() == 0:
            int_p_z_x += vicatsr.posterior_log_space(vicatsr._data, z,
                                                     exprs, ev)

        else:
            res, error = scipy.integrate.nquad(posterior_func,
                                               integration_bounds,
                                               args=(z, exprs, ev,
                                                     vicatsr))
            int_p_z_x += res

    return int_p_z_x


# Integrate prior w.r.t z, c and x.
# This should integrate to 1.0 if everything is working as expected
def integrate_prior(vicatsr, exprs):

    def prior_func(*args):

        c = args[:-2]
        vicatsr = args[-1]

        z = copy.copy(args[-2])
        z.set_distr_consts(c)

        return vicatsr._prior(z)

    int_prior = 0.0
    for z in exprs:

        # Create integration bounds for continuous parameters
        integration_bounds = [[-np.inf, np.inf]
                              for _ in range(z.num_distr_consts())]

        if z.num_distr_consts() == 0:
            int_prior += vicatsr._prior(z)

        else:

            out, error = scipy.integrate.nquad(
                prior_func,
                integration_bounds,
                args=(z, vicatsr)
            )
            int_prior += out

    return int_prior

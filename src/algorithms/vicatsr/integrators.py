import copy
import numpy as np
import scipy


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

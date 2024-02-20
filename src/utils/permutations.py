import numpy as np
np.set_printoptions(suppress=True)


# Recursive function that enumerates over the n-dimensional (where n is the
# number of float ranges "hyperlattice"?
def recursively_enumerate(param_index, param_instance, all_permutations, vals):

    if param_index == param_instance.size:

        all_permutations.append(np.copy(param_instance))
        return all_permutations

    else:

        for v in vals[param_index]:
            param_instance[param_index] = v
            all_permutations = recursively_enumerate(param_index + 1,
                                                     param_instance,
                                                     all_permutations,
                                                     vals)
        return all_permutations


# Compute all permutations of range of floats with given step sizes
def compute_permutations(lbs, ubs, step_sizes):

    vals = compute_ranges(lbs, ubs, step_sizes)

    all_permutations = []
    param_instance = np.zeros(len(vals), dtype=np.float64)
    recursively_enumerate(0, param_instance, all_permutations, vals)

    return np.vstack(all_permutations)


# Takes lower and upper bounds and step sizes as input and outputs full
# ranges of values
def compute_ranges(lbs, ubs, step_sizes):

    ranges = []

    for lb, ub, ss in zip(lbs, ubs, step_sizes):
        ranges.append(np.arange(lb, ub + ss, ss))

    return ranges

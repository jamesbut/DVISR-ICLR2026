# Network masks for softmax input.
# These are used to zero out certain token probabilities.

import torch
import numpy as np


class NetMasks:

    def __init__(self, token_set):

        self._token_set = token_set

        # A mask to apply so that only constants are sampled
        self._consts_mask = np.array(
            [0.0 if t['type'] == 'const' else -1e9 for t in token_set]
        )

        # A mask to apply so that only unary operators and consts are sampled
        self._un_ops_consts_mask = np.array(
            [0.0 if t['type'] == 'un_op' or t['type'] == 'const' else -1e9
             for t in token_set]
        )

        # A mask to turn off variable constants
        self._no_vars_mask = np.array(
            [-1e9 if t['sub_type'] == 'var_const' else 0.0
             for t in token_set]
        )

    # Compose mask from multiple mask names
    def compose_mask(self, mask_names):

        mask = np.zeros(len(self._token_set))

        if 'consts' or 'un_ops' in mask_names:

            # Only sample un_ops and consts
            if 'un_ops' in mask_names:
                mask = self._un_ops_consts_mask

            # Only sample consts
            else:
                mask = self._consts_mask

        if 'no_vars' in mask_names:

            # Turn off variables
            mask = mask + self._no_vars_mask

        return torch.from_numpy(mask)

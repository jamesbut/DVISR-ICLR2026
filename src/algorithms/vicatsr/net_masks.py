# Network masks for softmax input.
# These are used to zero out certain token probabilities.

import torch
import numpy as np
from typing import List


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

        # Masks to turn off variable constants
        self._no_var_masks = {}
        for t in token_set:
            if t['sub_type'] == 'var_const':
                mask = np.zeros(len(token_set))
                mask[t['id']] = -1e9
                self._no_var_masks[t['op']] = mask

    # Compose mask from multiple mask names
    # Can also remove variables by setting them in remove_vars
    def compose_mask(self, mask_names: List[str] = None,
                     remove_vars: List[str] = None):

        mask = np.zeros(len(self._token_set))

        if mask_names:

            if 'consts' or 'un_ops' in mask_names:

                # Only sample un_ops and consts
                if 'un_ops' in mask_names:
                    mask = self._un_ops_consts_mask

                # Only sample consts
                else:
                    mask = self._consts_mask

        # Turn off variables
        if remove_vars:
            for var in remove_vars:
                mask = mask + self._no_var_masks[var]

        return torch.from_numpy(mask)

# Network masks for softmax input.
# These are used to zero out certain token probabilities.

import torch
import numpy as np
from typing import List
from util.tree import is_descendent


class NetMasks:

    def __init__(self, token_set, constraints=None):

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

        # Mask to turn off log
        self._no_log_mask = np.array(
            [-1e9 if t['op'] == 'log' else 0.0 for t in token_set]
        )

        # Mask to turn off exp
        self._no_exp_mask = np.array(
            [-1e9 if t['op'] == 'exp' else 0.0 for t in token_set]
        )

        # Mask to turn off trig ops
        self._no_trig_mask = np.array(
            [-1e9 if t['op'] == 'sin' or t['op'] == 'cos' else 0.0
             for t in token_set]
        )

        # Masks to turn off variable constants
        self._no_var_masks = {}
        for t in token_set:
            if t['sub_type'] == 'var_const':
                mask = np.zeros(len(token_set))
                mask[t['id']] = -1e9
                self._no_var_masks[t['op']] = mask

        # List of all optional constraints to apply
        self._constraints = constraints

        # Check whether all specified constraints are valid
        if self._constraints:
            valid_constraints = [
                "inverse_ops",
                "nested_trigs"
            ]

            for c in self._constraints:
                if c not in valid_constraints:
                    raise ValueError(f'{c} not a valid constraint')

    # Determine the masks required at the current state of a sampling process
    def determine_masks(self,
                        max_num_tokens,
                        sampled_tokens,
                        num_consts_required):
        masks = []
        num_tokens_sampled = len(sampled_tokens)

        # Apply mask to only sample unary operators and constants
        if max_num_tokens - num_tokens_sampled == num_consts_required + 1:
            masks = ['un_ops', 'consts']

        # Apply mask to only sample constants
        if max_num_tokens - num_tokens_sampled < num_consts_required + 1:
            masks = ['consts']

        # Optional masks
        if self._constraints:

            # Mask for nested trig functions
            if 'nested_trigs' in self._constraints:
                if is_descendent(sampled_tokens, ['cos', 'sin']):
                    masks += ['no_trig']

            # Mask for log(exp()) and exp(log())
            if 'inverse_ops' in self._constraints and sampled_tokens:
                if sampled_tokens[-1]['op'] == 'log':
                    masks += ['no_exp']
                if sampled_tokens[-1]['op'] == 'exp':
                    masks += ['no_log']

        return masks

    # Compose mask from multiple mask names
    # Can also remove variables by setting them in remove_vars
    def compose_mask(self, mask_names: List[str] = None):

        if not mask_names:
            return None

        mask = np.zeros(len(self._token_set))

        if 'consts' in mask_names or 'un_ops' in mask_names:

            # Only sample un_ops and consts
            if 'un_ops' in mask_names:
                mask = self._un_ops_consts_mask.copy()

            # Only sample consts
            else:
                mask = self._consts_mask.copy()

        # Do not sample log
        if 'no_log' in mask_names:
            mask += self._no_log_mask

        # Do not sample exp
        if 'no_exp' in mask_names:
            mask += self._no_exp_mask

        # Do not sample trig ops
        if 'no_trig' in mask_names:
            mask += self._no_trig_mask

        return torch.from_numpy(mask)

    @property
    def constraints(self):
        return self._constraints

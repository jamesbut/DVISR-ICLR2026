# Network masks for softmax input.
# These are used to zero out certain token probabilities.

import torch
import numpy as np


class NetMasks:

    def __init__(self, token_set=None, j=None):

        if token_set:

            # A mask to apply so that only constants are sampled
            self._consts_mask = torch.from_numpy(np.array(
                [0.0 if t['type'] == 'const' else -1e9 for t in token_set]
            ))

            # A mask to apply so that only unary operators and consts are sampled
            self._un_ops_consts_mask = torch.from_numpy(np.array(
                [0.0 if t['type'] == 'un_op' or t['type'] == 'const' else -1e9
                 for t in token_set]
            ))

        if j:
            for key, value in j.items():
                setattr(self, "_" + key, torch.tensor(value))

    @property
    def consts_mask(self):
        return self._consts_mask

    @property
    def un_ops_consts_mask(self):
        return self._un_ops_consts_mask

    def to_json(self):

        j = {
            'consts_mask': self._consts_mask.tolist(),
            'un_ops_consts_mask': self._un_ops_consts_mask.tolist()
        }

        return j

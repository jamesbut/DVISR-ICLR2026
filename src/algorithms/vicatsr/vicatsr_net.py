import torch
torch.set_default_dtype(torch.float64)


class NN(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size,
                 distr_over_consts: bool, const_variance: bool,
                 init_gru_zero: bool):
        super().__init__()

        self._hidden_size = hidden_size

        self._l1 = None
        self._consts_layer = None

        if hidden_size == 0:

            self._l2 = torch.nn.Linear(num_inputs, num_outputs)

            if distr_over_consts:
                self._consts_layer = torch.nn.Linear(num_inputs, 2)

        else:
            self._l1 = torch.nn.GRUCell(num_inputs, self._hidden_size)
            self._l2 = torch.nn.Linear(self._hidden_size, num_outputs)

            if init_gru_zero:
                self._l1.apply(init_gru_weights_zero)

            if distr_over_consts:
                self._consts_layer = torch.nn.Linear(self._hidden_size, 2)

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        self._const_variance = const_variance

    def forward(self, x, pre_softmax_mask=None):

        # Check hidden state has been initialised
        if not hasattr(self, '_hx'):
            raise RuntimeError('Must call reset() before forward()')

        # GRU layer
        if self._l1 is not None:
            x = self._l1(x, self._hx)
            self._hx = x

        # Linear layer that produces logits for the categorical distribution
        cat_logits = self._l2(x)

        # Apply binary mask before the softmax - this is equivalent to
        # preventing some of the tokens being sampled
        # TODO: I do not know whether this should be in-place
        if pre_softmax_mask is not None:
            cat_logits += pre_softmax_mask

        # Softmax layer that converts logits to probabilities
        cat_params = torch.nn.functional.softmax(cat_logits, dim=0)
        output = cat_params

        # Output parameter of constant distribution
        if self._consts_layer is not None:
            const_out = self._consts_layer(x)

            if self._const_variance:
                const_out = torch.where(
                    torch.tensor([False, True]),
                    torch.nn.functional.softplus(const_out),
                    const_out
                )

            output = torch.cat((output, const_out), dim=0)

        return output

    def reset(self, batch_size):
        self._hx = torch.zeros(self._hidden_size)

    def num_inputs(self):
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs


import torch.nn.init as init


def init_gru_weights_zero(gru_cell):

    # init.xavier_uniform_(gru_cell.weight_ih)  # Input-to-hidden weights
    init.zeros_(gru_cell.weight_ih)  # Input-to-hidden weights
    init.orthogonal_(gru_cell.weight_hh)      # Hidden-to-hidden weights
    init.zeros_(gru_cell.bias_ih)             # Zero biases
    init.zeros_(gru_cell.bias_hh)

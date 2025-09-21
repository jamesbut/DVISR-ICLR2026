# Deep Variational Inference Symbolic Regression

This repository contains the code used to generate the results in the Deep Variational Inference Symbolic Regression paper submitted to ICLR 2026.
It also contains the specific results reported in the paper in the `results/` directory.

## Installation

A `requirements.txt` file has been provided with all the packages needed to run the code.
We recommend creating a virtual Python environment - using, for example, `venv` - and installing the required packages there.

## Results

The plots and results tables used in the paper can be displayed via the following commands.
For the no constant experiments for the quadratic, linear and constant data respectively:

```
python3 src/main.py -a results/quadratic/no_consts_prev_input --true_pos
python3 src/main.py -a results/linear/no_consts_prev_input --true_pos
python3 src/main.py -a results/constant/no_consts_prev_input --true_pos
```

For the constant experiments for the quadratic, linear and constant data respectively:

```
python3 src/main.py -a results/quadratic/consts_mt3_prev_input --true_pos
python3 src/main.py -a results/linear/consts_mt3_prev_input --true_pos
python3 src/main.py -a results/constant/consts_mt3_prev_input --true_pos
```

To analyse the results and produce the plot for the scaling experiments run the following command from the `scripts/` directory:

```
python3 kl_divergence_plot.py results/scaling/no_consts/prev_input/results
```

All of the above commands analyse the relevant results files in the `results/` directory.

## Experiments

To rerun the experiments from the paper copy the `config.json` file from the appropriate results directory into `configs/` and then run:

```
python3 src/main.py -c config.json
```

Once finished the results of the experiment will then be available in the results directory.

The corresponding config files for the no const experiments for the quadratic, linear and constant data respectively are:

```
results/quadratic/no_consts_prev_input/config.json
results/linear/no_consts_prev_input/config.json
results/constant/no_consts_prev_input/config.json
```

The corresponding config files for the const experiments for the quadratic, linear and constant data respectively are:

```
results/quadratic/consts_mt3_prev_input/config.json
results/linear/consts_mt3_prev_input/config.json
results/constant/consts_mt3_prev_input/config.json
```

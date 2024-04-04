from algorithms.algorithm import Algorithm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),
                '../../libs/MachineScientist'))

from libs.MachineScientist.parallel import Parallel
from libs.MachineScientist.Prior.fit_prior import read_prior_par

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class BayesianMachineScientist(Algorithm):

    def __init__(self, config):

        # Number of MCMC steps
        self._num_steps = config['num_steps']

        # Set the temperatures for the parallel tempering
        self._ts = [1] + [1.04**k for k in range(1, 20)]

        # Read parameters for prior over expressions from file
        self._prior_params = read_prior_par(
            os.getcwd()
            + '/libs/MachineScientist/Prior/'
            + 'final_prior_param_sq.named_equations.nv13'
            + '.np13.2016-09-01 17:05:57.196882.dat')

    def train(self, data):

        # Construct labels for all attributes
        x_labels = ['x%d' % i for i in range(data['x'].shape[1])]

        # Initialise the parallel machine scientist
        pms = Parallel(
            self._ts,
            variables=x_labels,
            parameters=['a%d' % i for i in range(data['x'].shape[1])],
            x=pd.DataFrame(data['x'], columns=x_labels),
            y=pd.Series(data['y']),
            prior_par=self._prior_params,
        )

        description_lengths, mdl, mdl_model = [], np.inf, None

        # Perform MCMC
        print('Performing Bayesian Machine Scientist MCMC...')
        for i in tqdm(range(self._num_steps)):

            # MCMC step within each temperature
            pms.mcmc_step()

            # Attempt to swap two randomly selected consecutive temperatures
            pms.tree_swap()

            # Add the description length to the trace
            description_lengths.append(pms.t1.E)

            # Check if this is the MDL expression so far
            if pms.t1.E < mdl:
                mdl, mdl_model = pms.t1.E, deepcopy(pms.t1)

        print('Best model:       ', mdl_model)
        print('Parameter values: ', mdl_model.par_values['d0'])
        print('Desc. length:     ', mdl)

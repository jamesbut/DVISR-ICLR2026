import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import unittest
from utils.json_helper import read_json, print_json
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm


class VICatSR(unittest.TestCase):

    def setUp(self):

        # Read config
        self._config = read_json(os.getcwd()
                                 + '/configs/test_configs/vicatsr.json')

        # Create domain
        self._domain = create_domain(self._config['domain'])
        self._data = self._domain.create_data()

    def test_max_likelihood_static_consts(self):

        self._config['algorithm']['operators']['consts'] = [1.0]
        self._config['algorithm']['max_likelihood'] = True

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, all_exps = alg.train(self._data)

        # q should converge to a one hot vector because it is a maximum
        # likelihood optimisation with all the weight on the highest likelihood
        # model, y = x
        self.assertLessEqual(q.pdf(all_exps[0]).item(), 0.01)
        self.assertGreaterEqual(q.pdf(all_exps[1]).item(), 0.99)

    def test_elbo_static_consts(self):

        self._config['algorithm']['operators']['consts'] = [1.0]

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, true_pos, all_exps = alg.train(self._data)

        # q should converge to the true posterior for both models
        self.assertAlmostEqual(q.pdf(all_exps[0]).item(), 0.06008665)
        self.assertAlmostEqual(true_pos[0], 0.06008665)

        self.assertAlmostEqual(q.pdf(all_exps[1]).item(), 0.939913349)
        self.assertAlmostEqual(true_pos[1], 0.939913349)

    def test_max_likelihood_opt_consts(self):

        self._config['algorithm']['operators']['consts'] = ['opt_const']
        self._config['algorithm']['max_likelihood'] = True

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, all_exps = alg.train(self._data)

        # q should still converge to a one hot vector even though constant
        # has been optimised
        self.assertLessEqual(q.pdf(all_exps[0]).item(), 0.01)
        self.assertGreaterEqual(q.pdf(all_exps[1]).item(), 0.99)

        # Check constant is optimised correctly
        self.assertAlmostEqual(all_exps[0].tokens()[0]['value'], 0.35, places=4)

    def test_elbo_opt_consts(self):

        self._config['algorithm']['operators']['consts'] = ['opt_const']

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, true_pos, all_exps = alg.train(self._data)

        # q should converge to the true posterior for both models
        # It is different to that with static constants because the optimised
        # y = c is more likely than y = 1.0
        self.assertAlmostEqual(q.pdf(all_exps[0]).item(), 0.39502215)
        self.assertAlmostEqual(true_pos[0], 0.39502215)

        self.assertAlmostEqual(q.pdf(all_exps[1]).item(), 0.60497784)
        self.assertAlmostEqual(true_pos[1], 0.60497784)

    # TODO: Test max likelihood with distr consts
    # TODO: Test ELBO with distr consts


if __name__ == "__main__":
    unittest.main()

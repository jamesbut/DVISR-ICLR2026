import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import unittest
from utils.json_helper import read_json
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm
from algorithms.vicatsr.equation import Equation
from utils.tree import get_parent, get_sibling


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
        self._config['algorithm']['learning_rate'] = 1e-3

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
        self._config['algorithm']['learning_rate'] = 1e-3

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, true_pos, all_exps = alg.train(self._data)

        # q should converge to the true posterior for both models
        self.assertAlmostEqual(q.pdf(all_exps[0]).item(), true_pos[0], places=5)
        self.assertAlmostEqual(q.pdf(all_exps[1]).item(), true_pos[1], places=5)

    def test_max_likelihood_opt_consts(self):

        self._config['algorithm']['operators']['consts'] = ['opt_const']
        self._config['algorithm']['max_likelihood'] = True
        self._config['algorithm']['learning_rate'] = 1e-3

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
        self._config['algorithm']['learning_rate'] = 1e-3

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, true_pos, all_exps = alg.train(self._data)

        # q should converge to the true posterior for both models
        # It is different to that with static constants because the optimised
        # y = c is more likely than y = 1.0
        self.assertAlmostEqual(q.pdf(all_exps[0]).item(), 0.39502215, places=3)
        self.assertAlmostEqual(true_pos[0], 0.39502215, places=3)

        self.assertAlmostEqual(q.pdf(all_exps[1]).item(), 0.60497784, places=3)
        self.assertAlmostEqual(true_pos[1], 0.60497784, places=3)

    def test_max_likelihood_distr_consts(self):

        self._config['algorithm']['max_likelihood'] = True
        self._config['algorithm']['learning_rate'] = 1e-3

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, all_exps = alg.train(self._data)

        # q should still converge to a one hot vector
        self.assertLessEqual(q.pdf(all_exps[0]).item(), 0.05)
        self.assertGreaterEqual(q.pdf(all_exps[1]).item(), 0.95)

    def test_elbo_distr_consts_no_x(self):

        self._config['algorithm']['learning_rate'] = 2e-4
        self._config['algorithm']['remove_x_vars'] = True

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, true_pos, all_exps = alg.train(self._data)

        # Get const mean and variance
        consts_params = q.get_consts_params(all_exps[0])
        mean = consts_params[0][0]
        variance = consts_params[0][1]

        print('Mean:', mean)
        print('Variance:', variance)

        # Check the parameters for the distribution over constants has
        # optimised
        self.assertLessEqual(mean, 0.37)
        self.assertGreaterEqual(mean, 0.33)
        self.assertLessEqual(variance, 0.32)
        self.assertGreaterEqual(variance, 0.28)

    def test_elbo_distr_consts_separate_behaviour_policy(self):

        self._config['algorithm']['learning_rate'] = 2e-4
        self._config['algorithm']['behaviour_policy'] = 'equal_prob_tokens'
        self._config['algorithm']['num_eq_samples'] = 100
        self._config['algorithm']['plotting'] = False

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'])

        # Train
        q, true_pos, all_exps = alg.train(self._data)

        # Get const mean and variance
        consts_params = q.get_consts_params(all_exps[0])
        mean = consts_params[0][0]
        variance = consts_params[0][1]

        print('Mean:', mean)
        print('Variance:', variance)

        # Check the parameters for the distribution over constants has
        # optimised
        self.assertLessEqual(mean, 0.37)
        self.assertGreaterEqual(mean, 0.33)
        self.assertLessEqual(variance, 0.32)
        self.assertGreaterEqual(variance, 0.28)


class Utils(unittest.TestCase):

    def test_equation(self):

        # Test equation can be created from a token set and an infix string
        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 1},
            {'op': '*', 'type': 'bin_op', 'sub_type': None, 'id': 2},
            {'op': 'sin', 'type': 'un_op', 'sub_type': None, 'id': 3},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 4}
        ]

        eq1 = Equation(infix_str='(x_0 * x_0) + x_0', token_set=token_set)
        self.assertEqual(eq1.get_infix(), '((x_0 * x_0) + x_0)')

        eq2 = Equation(infix_str='sin(x_0)', token_set=token_set)
        self.assertEqual(eq2.get_infix(), 'sin(x_0)')

        eq3 = Equation(infix_str='(sin((x_0 * x_0)) + x_0)', token_set=token_set)
        self.assertEqual(eq3.get_infix(), '(sin((x_0 * x_0)) + x_0)')

    def test_get_parent(self):

        # Test get_parent function with respect to an Equation
        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 1},
            {'op': '*', 'type': 'bin_op', 'sub_type': None, 'id': 2},
            {'op': 'sin', 'type': 'un_op', 'sub_type': None, 'id': 3},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 4}
        ]

        eq1 = Equation(infix_str='(x_0 + x_0)', token_set=token_set)
        self.assertEqual(get_parent(eq1.tokens())['op'], '+')
        self.assertEqual(get_parent(eq1.tokens()[:-1])['op'], '+')
        self.assertEqual(get_parent(eq1.tokens()[:-2]), None)

        eq2 = Equation(infix_str='((sin(x_0) * x_0) + x_0)', token_set=token_set)
        self.assertEqual(get_parent(eq2.tokens()[:1]), None)
        self.assertEqual(get_parent(eq2.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq2.tokens()[:3])['op'], '*')
        self.assertEqual(get_parent(eq2.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_parent(eq2.tokens()[:5])['op'], '*')
        self.assertEqual(get_parent(eq2.tokens())['op'], '+')

        eq3 = Equation(infix_str='(sin(sin(x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_parent(eq3.tokens()[:1]), None)
        self.assertEqual(get_parent(eq3.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq3.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq3.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_parent(eq3.tokens())['op'], '+')

        eq4 = Equation(infix_str='(sin((x_0 * x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_parent(eq4.tokens()[:1]), None)
        self.assertEqual(get_parent(eq4.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq4.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq4.tokens()[:4])['op'], '*')
        self.assertEqual(get_parent(eq4.tokens()[:5])['op'], '*')
        self.assertEqual(get_parent(eq4.tokens())['op'], '+')

        eq5 = Equation(infix_str='(sin(x_0) + sin(x_0))', token_set=token_set)
        self.assertEqual(get_parent(eq5.tokens()[:1]), None)
        self.assertEqual(get_parent(eq5.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq5.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq5.tokens()[:4])['op'], '+')
        self.assertEqual(get_parent(eq5.tokens())['op'], 'sin')

        eq6 = Equation(infix_str='(x_0 + sin(sin(x_0)))', token_set=token_set)
        self.assertEqual(get_parent(eq6.tokens()[:1]), None)
        self.assertEqual(get_parent(eq6.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq6.tokens()[:3])['op'], '+')
        self.assertEqual(get_parent(eq6.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_parent(eq6.tokens())['op'], 'sin')

        eq7 = Equation(infix_str='(sin(x_0) + (x_0 * x_0))', token_set=token_set)
        self.assertEqual(get_parent(eq7.tokens()[:1]), None)
        self.assertEqual(get_parent(eq7.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq7.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq7.tokens()[:4])['op'], '+')
        self.assertEqual(get_parent(eq7.tokens()[:5])['op'], '*')
        self.assertEqual(get_parent(eq7.tokens())['op'], '*')

    def test_get_sibling(self):

        # Test get_sibling function with respect to an Equation
        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 1},
            {'op': '*', 'type': 'bin_op', 'sub_type': None, 'id': 2},
            {'op': 'sin', 'type': 'un_op', 'sub_type': None, 'id': 3},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 4}
        ]

        eq1 = Equation(infix_str='(x_0 + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq1.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq1.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq1.tokens())['op'], 'x_0')

        eq2 = Equation(infix_str='(sin(x_0) + sin(x_0))', token_set=token_set)
        self.assertEqual(get_sibling(eq2.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq2.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq2.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq2.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_sibling(eq2.tokens()[:5]), None)

        eq3 = Equation(infix_str='(sin((x_0 * x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq3.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:5])['op'], 'x_0')
        self.assertEqual(get_sibling(eq3.tokens())['op'], 'sin')

        eq4 = Equation(infix_str='((sin(x_0) * x_0) + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq4.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:5])['op'], 'sin')
        self.assertEqual(get_sibling(eq4.tokens())['op'], '*')

        eq5 = Equation(infix_str='(sin(sin(x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq5.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq5.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq5.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq5.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq5.tokens())['op'], 'sin')

        eq6 = Equation(infix_str='(x_0 + sin(sin(x_0)))', token_set=token_set)
        self.assertEqual(get_sibling(eq6.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq6.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq6.tokens()[:3])['op'], 'x_0')
        self.assertEqual(get_sibling(eq6.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq6.tokens()), None)

        eq7 = Equation(infix_str='(sin(x_0) + (x_0 * x_0))', token_set=token_set)
        self.assertEqual(get_sibling(eq7.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq7.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq7.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq7.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_sibling(eq7.tokens()[:5]), None)
        self.assertEqual(get_sibling(eq7.tokens())['op'], 'x_0')


if __name__ == "__main__":
    unittest.main()

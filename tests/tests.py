import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import unittest
from util.json_helper import read_json
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm
from algorithms.vicatsr.equation import Equation
from algorithms.vicatsr.net_masks import NetMasks
from algorithms.vicatsr.vicatsr import likelihood
from util.tree import get_parent, get_sibling, is_descendent
import torch
from scipy.stats import norm
import numpy as np
import math
import copy
import scipy


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
        alg = create_algorithm(self._config['algorithm'], self._domain)

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
        alg = create_algorithm(self._config['algorithm'], self._domain)

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
        alg = create_algorithm(self._config['algorithm'], self._domain)

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
        alg = create_algorithm(self._config['algorithm'], self._domain)

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
        alg = create_algorithm(self._config['algorithm'], self._domain)

        # Train
        q, all_exps = alg.train(self._data)

        # q should still converge to a one hot vector
        self.assertLessEqual(q.pdf(all_exps[0]).item(), 0.05)
        self.assertGreaterEqual(q.pdf(all_exps[1]).item(), 0.95)

    def test_elbo_distr_consts_no_x(self):

        self._config['algorithm']['learning_rate'] = 2e-4
        self._config['algorithm']['remove_x_vars'] = True

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'], self._domain)

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
        alg = create_algorithm(self._config['algorithm'], self._domain)

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

    def test_parent_sibling_input(self):

        self._config['algorithm']['max_num_tokens'] = 5
        self._config['algorithm']['num_steps'] = 5
        self._config['algorithm']['distr_over_consts'] = False
        self._config['algorithm']['max_likelihood'] = True

        self._config['algorithm']['target_policy']['parent_input'] = True
        self._config['algorithm']['target_policy']['sibling_input'] = True

        # Create algoritm
        alg = create_algorithm(self._config['algorithm'], self._domain)

        # Train
        alg.train(self._data)


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
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq1.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq1.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq1.tokens()), None)

        eq2 = Equation(infix_str='((sin(x_0) * x_0) + x_0)', token_set=token_set)
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq2.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq2.tokens()[:2])['op'], '*')
        self.assertEqual(get_parent(eq2.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq2.tokens()[:4])['op'], '*')
        self.assertEqual(get_parent(eq2.tokens()[:5])['op'], '+')
        self.assertEqual(get_parent(eq2.tokens()), None)

        eq3 = Equation(infix_str='(sin(sin(x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq3.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq3.tokens()[:2])['op'], 'sin')
        self.assertEqual(get_parent(eq3.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq3.tokens()[:4])['op'], '+')
        self.assertEqual(get_parent(eq3.tokens()), None)

        eq4 = Equation(infix_str='(sin((x_0 * x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq4.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq4.tokens()[:2])['op'], 'sin')
        self.assertEqual(get_parent(eq4.tokens()[:3])['op'], '*')
        self.assertEqual(get_parent(eq4.tokens()[:4])['op'], '*')
        self.assertEqual(get_parent(eq4.tokens()[:5])['op'], '+')
        self.assertEqual(get_parent(eq4.tokens()), None)

        eq5 = Equation(infix_str='(sin(x_0) + sin(x_0))', token_set=token_set)
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq5.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq5.tokens()[:2])['op'], 'sin')
        self.assertEqual(get_parent(eq5.tokens()[:3])['op'], '+')
        self.assertEqual(get_parent(eq5.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_parent(eq5.tokens()), None)

        eq6 = Equation(infix_str='(x_0 + sin(sin(x_0)))', token_set=token_set)
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq6.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq6.tokens()[:2])['op'], '+')
        self.assertEqual(get_parent(eq6.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_parent(eq6.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_parent(eq6.tokens()), None)

        eq7 = Equation(infix_str='(sin(x_0) + (x_0 * x_0))', token_set=token_set)
        self.assertEqual(get_parent([]), None)
        self.assertEqual(get_parent(eq7.tokens()[:1])['op'], '+')
        self.assertEqual(get_parent(eq7.tokens()[:2])['op'], 'sin')
        self.assertEqual(get_parent(eq7.tokens()[:3])['op'], '+')
        self.assertEqual(get_parent(eq7.tokens()[:4])['op'], '*')
        self.assertEqual(get_parent(eq7.tokens()[:5])['op'], '*')
        self.assertEqual(get_parent(eq7.tokens()), None)

    def test_get_sibling(self):

        # Test get_sibling function with respect to an Equation
        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 1},
            {'op': '*', 'type': 'bin_op', 'sub_type': None, 'id': 2},
            {'op': 'sin', 'type': 'un_op', 'sub_type': None, 'id': 3},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 4}
        ]

        eq1 = Equation(infix_str='(x_0 + x_0)', token_set=token_set)
        self.assertEqual(get_sibling([]), None)
        self.assertEqual(get_sibling(eq1.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq1.tokens()[:2])['op'], 'x_0')
        self.assertEqual(get_sibling(eq1.tokens()), None)

        eq2 = Equation(infix_str='(sin(x_0) + sin(x_0))', token_set=token_set)
        self.assertEqual(get_sibling(eq2.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq2.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq2.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_sibling(eq2.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq2.tokens()), None)

        eq3 = Equation(infix_str='(sin((x_0 * x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq3.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq3.tokens()[:4])['op'], 'x_0')
        self.assertEqual(get_sibling(eq3.tokens()[:5])['op'], 'sin')
        self.assertEqual(get_sibling(eq3.tokens()), None)

        eq4 = Equation(infix_str='((sin(x_0) * x_0) + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq4.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq4.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_sibling(eq4.tokens()[:5])['op'], '*')
        self.assertEqual(get_sibling(eq4.tokens()), None)

        eq5 = Equation(infix_str='(sin(sin(x_0)) + x_0)', token_set=token_set)
        self.assertEqual(get_sibling(eq5.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq5.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq5.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq5.tokens()[:4])['op'], 'sin')
        self.assertEqual(get_sibling(eq5.tokens()), None)

        eq6 = Equation(infix_str='(x_0 + sin(sin(x_0)))', token_set=token_set)
        self.assertEqual(get_sibling(eq6.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq6.tokens()[:2])['op'], 'x_0')
        self.assertEqual(get_sibling(eq6.tokens()[:3]), None)
        self.assertEqual(get_sibling(eq6.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq6.tokens()), None)

        eq7 = Equation(infix_str='(sin(x_0) + (x_0 * x_0))', token_set=token_set)
        self.assertEqual(get_sibling(eq7.tokens()[:1]), None)
        self.assertEqual(get_sibling(eq7.tokens()[:2]), None)
        self.assertEqual(get_sibling(eq7.tokens()[:3])['op'], 'sin')
        self.assertEqual(get_sibling(eq7.tokens()[:4]), None)
        self.assertEqual(get_sibling(eq7.tokens()[:5])['op'], 'x_0')
        self.assertEqual(get_sibling(eq7.tokens()), None)

    def test_equation_sympy(self):

        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 1},
            {'op': '*', 'type': 'bin_op', 'sub_type': None, 'id': 2},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 3}
        ]

        eq1 = Equation(infix_str='((x_0 * x_0) + x_0) + x_0', token_set=token_set)
        self.assertEqual(eq1.get_infix(), '(((x_0 * x_0) + x_0) + x_0)')
        self.assertEqual(eq1.get_infix(simplify=True), 'x_0**2 + 2*x_0')

    def test_equation_equality(self):

        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 1},
            {'op': '-', 'type': 'bin_op', 'sub_type': None, 'id': 2},
            {'op': '*', 'type': 'bin_op', 'sub_type': None, 'id': 3},
            {'op': '/', 'type': 'bin_op', 'sub_type': None, 'id': 4},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 5}
        ]

        # Confirm that expansion of the following equation string is equal
        # to Nguyen-4
        eq_str = ('((x_0 * ((x_0 / x_0) + x_0)) * '
                  '(((x_0 / x_0) - x_0) + (x_0 * x_0))) * '
                  '(((x_0 / x_0) + x_0) + (x_0 * x_0))')
        eq = Equation(infix_str=eq_str, token_set=token_set)
        self.assertEqual(
            eq.get_infix(True),
            'x_0**6 + x_0**5 + x_0**4 + x_0**3 + x_0**2 + x_0'
        )
        self.assertEqual(eq.num_tokens(), 27)

    def test_is_descendent(self):

        eq_1 = [
            {'op': 'cos', 'type': 'un_op'}
        ]

        self.assertTrue(is_descendent(eq_1, ['cos']))
        self.assertFalse(is_descendent(eq_1, ['sin']))

        eq_2 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'cos', 'type': 'un_op'},
        ]

        self.assertTrue(is_descendent(eq_2, ['cos']))
        self.assertFalse(is_descendent(eq_2, ['sin']))

        eq_3 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'cos', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'}
        ]

        self.assertFalse(is_descendent(eq_3, ['cos']))
        self.assertFalse(is_descendent(eq_3, ['sin']))

        eq_4 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'cos', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'},
            {'op': 'sin', 'type': 'un_op'},
        ]

        self.assertFalse(is_descendent(eq_4, ['cos']))
        self.assertTrue(is_descendent(eq_4, ['sin']))

        eq_5 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': '*', 'type': 'bin_op'},
            {'op': 'x_0', 'type': 'const'},
            {'op': 'cos', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'},
            {'op': 'sin', 'type': 'un_op'},
        ]

        self.assertFalse(is_descendent(eq_5, ['cos']))
        self.assertTrue(is_descendent(eq_5, ['sin']))

    def test_net_masks(self):

        token_set = [
            {'op': '+', 'type': 'bin_op', 'sub_type': None, 'id': 0},
            {'op': 'cos', 'type': 'un_op', 'sub_type': None, 'id': 1},
            {'op': 'sin', 'type': 'un_op', 'sub_type': None, 'id': 2},
            {'op': 'exp', 'type': 'un_op', 'sub_type': None, 'id': 3},
            {'op': 'log', 'type': 'un_op', 'sub_type': None, 'id': 4},
            {'op': 'x_0', 'type': 'const', 'sub_type': 'var_const', 'id': 5}
        ]

        net_masks = NetMasks(token_set, ['inverse_ops', 'nested_trigs'])

        eq_1 = [
            {'op': 'cos', 'type': 'un_op'}
        ]

        # Check no trig mask works
        masks = net_masks.determine_masks(
            max_num_tokens=10, sampled_tokens=eq_1, num_consts_required=1
        )
        self.assertEqual(masks, ['no_trig'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([0.0, -1e9, -1e9, 0.0, 0.0, 0.0])
        ))

        # Check no trig mask is not included if not specified in constraints
        net_masks = NetMasks(token_set, ['inverse_ops'])

        masks = net_masks.determine_masks(
            max_num_tokens=10, sampled_tokens=eq_1, num_consts_required=1
        )
        self.assertEqual(masks, [])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertEqual(pre_softmax_mask, None)

        # Check the same for sin
        eq_2 = [
            {'op': 'sin', 'type': 'un_op'}
        ]

        net_masks = NetMasks(token_set, ['inverse_ops', 'nested_trigs'])

        masks = net_masks.determine_masks(
            max_num_tokens=10, sampled_tokens=eq_2, num_consts_required=1
        )
        self.assertEqual(masks, ['no_trig'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([0.0, -1e9, -1e9, 0.0, 0.0, 0.0])
        ))

        # Check for more complicated tree
        eq_3 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'sin', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'}
        ]

        masks = net_masks.determine_masks(
            max_num_tokens=10, sampled_tokens=eq_3, num_consts_required=1
        )
        self.assertEqual(masks, [])

        # Check for inverse_ops constraint
        eq_4 = [
            {'op': 'exp', 'type': 'un_op'}
        ]

        net_masks = NetMasks(token_set, ['inverse_ops', 'nested_trigs'])

        masks = net_masks.determine_masks(
            max_num_tokens=10, sampled_tokens=eq_4, num_consts_required=1
        )
        self.assertEqual(masks, ['no_log'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([0.0, 0.0, 0.0, 0.0, -1e9, 0.0])
        ))

        eq_5 = [
            {'op': 'log', 'type': 'un_op'}
        ]

        net_masks = NetMasks(token_set, ['inverse_ops', 'nested_trigs'])

        masks = net_masks.determine_masks(
            max_num_tokens=10, sampled_tokens=eq_5, num_consts_required=1
        )
        self.assertEqual(masks, ['no_exp'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([0.0, 0.0, 0.0, -1e9, 0.0, 0.0])
        ))

        # Check for const and un_ops masks with trig mask
        eq_6 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'sin', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'},
            {'op': 'cos', 'type': 'un_op'},
        ]

        masks = net_masks.determine_masks(
            max_num_tokens=5, sampled_tokens=eq_6, num_consts_required=1
        )
        self.assertEqual(masks, ['consts', 'no_trig'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([-1e9, -2e9, -2e9, -1e9, -1e9, 0.0])
        ))

        masks = net_masks.determine_masks(
            max_num_tokens=6, sampled_tokens=eq_6, num_consts_required=1
        )
        self.assertEqual(masks, ['un_ops', 'consts', 'no_trig'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([-1e9, -1e9, -1e9, 0.0, 0.0, 0.0])
        ))

        # Check for const and un_ops masks with inverse ops masks
        eq_7 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'exp', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'},
            {'op': 'log', 'type': 'un_op'},
        ]

        masks = net_masks.determine_masks(
            max_num_tokens=5, sampled_tokens=eq_7, num_consts_required=1
        )
        self.assertEqual(masks, ['consts', 'no_exp'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([-1e9, -1e9, -1e9, -2e9, -1e9, 0.0])
        ))

        masks = net_masks.determine_masks(
            max_num_tokens=6, sampled_tokens=eq_7, num_consts_required=1
        )
        self.assertEqual(masks, ['un_ops', 'consts', 'no_exp'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([-1e9, 0.0, 0.0, -1e9, 0.0, 0.0])
        ))

        eq_8 = [
            {'op': '+', 'type': 'bin_op'},
            {'op': 'log', 'type': 'un_op'},
            {'op': 'x_0', 'type': 'const'},
            {'op': 'exp', 'type': 'un_op'},
        ]

        masks = net_masks.determine_masks(
            max_num_tokens=5, sampled_tokens=eq_8, num_consts_required=1
        )
        self.assertEqual(masks, ['consts', 'no_log'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([-1e9, -1e9, -1e9, -1e9, -2e9, 0.0])
        ))

        masks = net_masks.determine_masks(
            max_num_tokens=6, sampled_tokens=eq_8, num_consts_required=1
        )
        self.assertEqual(masks, ['un_ops', 'consts', 'no_log'])

        pre_softmax_mask = net_masks.compose_mask(masks)
        self.assertTrue(torch.equal(
            pre_softmax_mask,
            torch.tensor([-1e9, 0.0, 0.0, 0.0, -1e9, 0.0])
        ))


class Reachability(unittest.TestCase):

    def setUp(self):

        # Read config
        self._config = read_json(os.getcwd()
                                 + '/configs/test_configs/vicatsr.json')

        self._config['algorithm']['target_policy']['parent_input'] = True
        self._config['algorithm']['target_policy']['sibling_input'] = True
        self._config['algorithm']['target_policy']['previous_input'] = False

        self._config['domain'] = {
            'name': 'DSOBenchmarks'
        }

    # Test reachability of Nguyen problems under certain configs
    def test_nguyen(self):

        self._config['algorithm']['operators']['binary_ops'] = [
            '+', '-', '*', '/'
        ]
        self._config['algorithm']['operators']['unary_ops'] = [
            'cos', 'sin', 'exp', 'log'
        ]
        self._config['algorithm']['max_num_tokens'] = 30

        # Create domain
        self._config['domain']['dataset'] = 'Nguyen-1'
        domain = create_domain(self._config['domain'])
        data = domain.create_data()

        # Create algorithm
        alg = create_algorithm(self._config['algorithm'], domain)
        alg._initialise(data)

        # Check simple equation is reachable
        eq1_str = '(x_0 * x_0)'
        eq1 = Equation(infix_str=eq1_str, token_set=alg._token_set)
        eq1.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertTrue(alg._q.pdf(eq1) > 0.0)

        # Check Nguyen-1 is reachable
        eq2_str = '((x_0 * (x_0 * x_0)) + (x_0 * x_0)) + x_0'
        eq2 = Equation(infix_str=eq2_str, token_set=alg._token_set)
        eq2.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq2.get_infix(True), 'x_0**3 + x_0**2 + x_0')
        self.assertTrue(alg._q.pdf(eq2) > 0.0)

        # Check Nguyen-2 is reachable
        eq3_str = ('(x_0 * (x_0 * (x_0 * x_0))) + (((x_0 * (x_0 * x_0)) + '
                   '(x_0 * x_0)) + x_0)')
        eq3 = Equation(infix_str=eq3_str, token_set=alg._token_set)
        eq3.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq3.get_infix(True), 'x_0**4 + x_0**3 + x_0**2 + x_0')
        self.assertTrue(alg._q.pdf(eq3) > 0.0)

        # Check Nguyen-3 is reachable
        eq4_str = ('(x_0 * (x_0 * (x_0 * (x_0 * x_0)))) + '
                   '((x_0 * (x_0 * (x_0 * x_0))) + (((x_0 * (x_0 * x_0)) + '
                   '(x_0 * x_0)) + x_0))')
        eq4 = Equation(infix_str=eq4_str, token_set=alg._token_set)
        eq4.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq4.get_infix(True),
                         'x_0**5 + x_0**4 + x_0**3 + x_0**2 + x_0')
        self.assertTrue(alg._q.pdf(eq4) > 0.0)

        # Check Nguyen-4 is not reachable in this form (not enough tokens)
        eq5_str = ('(x_0 * (x_0 * (x_0 * (x_0 * (x_0 * x_0))))) + '
                   '((x_0 * (x_0 * (x_0 * (x_0 * x_0)))) + '
                   '((x_0 * (x_0 * (x_0 * x_0))) + (((x_0 * (x_0 * x_0)) + '
                   '(x_0 * x_0)) + x_0)))')
        eq5 = Equation(infix_str=eq5_str, token_set=alg._token_set)
        eq5.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq5.get_infix(True),
                         'x_0**6 + x_0**5 + x_0**4 + x_0**3 + x_0**2 + x_0')
        self.assertTrue(alg._q.pdf(eq5) == 0.0)

        # But Nguyen-4 is reachable in this form
        eq6_str = ('((x_0 * ((x_0 / x_0) + x_0)) * '
                   '(((x_0 / x_0) - x_0) + (x_0 * x_0))) * '
                   '(((x_0 / x_0) + x_0) + (x_0 * x_0))')
        eq6 = Equation(infix_str=eq6_str, token_set=alg._token_set)
        eq6.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq6.get_infix(True),
                         'x_0**6 + x_0**5 + x_0**4 + x_0**3 + x_0**2 + x_0')
        self.assertTrue(alg._q.pdf(eq6) > 0.0)

        # Check Nguyen-5 is reachable
        eq7_str = '(sin((x_0 * x_0)) * cos(x_0)) - (x_0 / x_0)'
        eq7 = Equation(infix_str=eq7_str, token_set=alg._token_set)
        eq7.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq7.get_infix(True), 'sin(x_0**2)*cos(x_0) - 1')
        self.assertTrue(alg._q.pdf(eq7) > 0.0)

        # Check Nguyen-6 is reachable
        eq8_str = 'sin(x_0) + sin(((x_0 * x_0) + x_0))'
        eq8 = Equation(infix_str=eq8_str, token_set=alg._token_set)
        eq8.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq8.get_infix(True), 'sin(x_0) + sin(x_0**2 + x_0)')
        self.assertTrue(alg._q.pdf(eq8) > 0.0)

        # Check Nguyen-7 is reachable
        eq9_str = 'log(x_0 + (x_0 / x_0)) + log((x_0 * x_0) + (x_0 / x_0))'
        eq9 = Equation(infix_str=eq9_str, token_set=alg._token_set)
        eq9.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq9.get_infix(True), 'log(x_0 + 1) + log(x_0**2 + 1)')
        self.assertTrue(alg._q.pdf(eq9) > 0.0)

        # Check Nguyen-8 is reachable
        eq10_str = 'exp((x_0 / (x_0 + x_0)) * log(x_0))'
        eq10 = Equation(infix_str=eq10_str, token_set=alg._token_set)
        eq10.apply_pre_softmax_mask(alg._max_num_tokens, alg._net_masks)
        self.assertEqual(eq10.get_infix(True), 'sqrt(x_0)')
        self.assertTrue(alg._q.pdf(eq10) > 0.0)


class Integrator(unittest.TestCase):

    def setUp(self):

        # Read config
        self._config = read_json(os.getcwd()
                                 + '/configs/test_configs/vicatsr.json')

        self._config['domain'] = {
            "name": "WrittenExpression",
            "expression": "`x_0` * `x_0`",
            "x_mins": [0.0],
            "x_maxs": [1.0],
            "x_step_sizes": [0.1]
        }
        self._config['algorithm']['distr_over_consts'] = True
        self._config['algorithm']['binary_ops'] = ['+', '*']
        self._config['algorithm']['prior_variance'] = 0.1

        # Create domain
        self._domain = create_domain(self._config['domain'])
        self._data = self._domain.create_data()

        # Create algorithm
        self._alg = create_algorithm(self._config['algorithm'], self._domain)
        self._alg._initialise(self._data)

    def test_evidence(self):

        all_exprs = self._alg._enumerate_expressions(self._data)

        # Calculate log evidence without splitting the sum out of the numerical
        # integration
        log_evidence_no_split_sum = self._alg.log_evidence(
            self._data,
            all_exprs,
            int_method='no_split_sum'
        )

        self.assertAlmostEqual(log_evidence_no_split_sum, -10.485845747449712,
                               places=8)

        # Calculate log evidence by splitting the sum over expressions out of
        # the numerical integration
        log_evidence_split_sum = self._alg.log_evidence(
            self._data,
            all_exprs,
            int_method='split_sum',
            reset=True
        )

        # Check these evidence values are the same
        self.assertAlmostEqual(log_evidence_no_split_sum,
                               log_evidence_split_sum,
                               places=8)

        # Calculate log evidence by splitting the sum over expressions and
        # only integrating over the c values in the particular equation
        log_evidence_only_own_c = self._alg.log_evidence(
            self._data,
            all_exprs,
            int_method='only_own_c',
            reset=True
        )

        # Check these evidence values are the same
        self.assertAlmostEqual(log_evidence_no_split_sum,
                               log_evidence_only_own_c,
                               places=8)

        # Check posterior integrates to 1
        def post(ev, z, data, alg):
            return (likelihood(data, z, alg._max_num_tokens, alg._net_masks)
                    * alg._prior(z) / ev)

        def post_func(*args):

            z = copy.copy(args[1])
            c = args[0]
            data = args[2]
            ev = args[3]
            alg = args[4]

            if z.num_distr_consts() == 0:
                if c < 0.0 or c > 1.0:
                    return 0.0
            else:
                z.set_distr_consts([c])

            return post(ev, z, data, alg)

        def int_post(ev):
            integration_bounds = [[-np.inf, np.inf]]

            int_p_z_x = 0.0
            for z in all_exprs:
                out, error = scipy.integrate.nquad(
                    post_func,
                    integration_bounds,
                    args=(z,
                          self._data,
                          ev,
                          self._alg)
                )
                int_p_z_x += out

            return int_p_z_x

        int_post_numeric = int_post(np.exp(log_evidence_split_sum))

        self.assertAlmostEqual(int_post_numeric, 1.0, places=10)

        # Check prior integrates to 1
        def prior_func(*args):

            z = copy.copy(args[1])
            c = args[0]
            alg = args[2]

            if z.num_distr_consts() == 0:
                if c < 0.0 or c > 1.0:
                    return 0.0
            else:
                z.set_distr_consts([c])

            return alg._prior(z)

        int_prior = 0.0
        for z in all_exprs:
            out, error = scipy.integrate.nquad(
                prior_func,
                [[-np.inf, np.inf]],
                args=(z, self._alg)
            )
            int_prior += out

        self.assertAlmostEqual(int_prior, 1.0, places=10)

        # Calculate log evidence analytically
        # NOTE: I tried to do this but I could not get the calculated
        # evidence to match the evidence calculated by the numerical integrator

        '''
        # First expression y = x
        z1 = all_exprs[1]
        # Second expression y = c
        z2 = all_exprs[0]
        z2.set_distr_consts([self._config['algorithm']['prior_mean']])

        prior_std = self._config['algorithm']['prior_variance']
        ll_std = 1.0

        p_z1 = 0.5
        p_z2 = 0.5
        p_x_z1 = likelihood(self._data, z1, self._alg._max_num_tokens,
                            self._alg._net_masks)
        joint_z1 = p_z1 * p_x_z1

        y_mean = np.mean(self._data['y'])

        y_spread = math.prod(norm.pdf(self._data['y'], y_mean, ll_std))

        x_mean_l_var = prior_std ** 2 + (ll_std ** 2 / len(self._data['y']))
        x_mean_l = norm.pdf(
            y_mean,
            self._config['algorithm']['prior_mean'],
            np.sqrt(x_mean_l_var)
        )

        p_x_z2 = x_mean_l * y_spread

        joint_z2 = p_z2 * p_x_z2

        p_x_analytic = joint_z1 + joint_z2

        print('joint z1:', joint_z1)
        print('joint z2:', joint_z2)
        print('x_mean_l_var:', x_mean_l_var)
        print('p(x|z1):', p_x_z1)
        print('p(z1):', p_z1)
        print('p(z2):', p_z2)
        print('x_mean_l:', x_mean_l)
        print('y_spread:', y_spread)
        print('p_x_z2:', p_x_z2)

        print('p(x):', p_x_analytic)
        print('log(p(x)):', np.log(p_x_analytic))
        print('numerical p(x):', log_evidence_split_sum)

        int_post_analytic = int_post(p_x_analytic)
        print('Int post analytic:', int_post_analytic)
        '''


if __name__ == "__main__":
    unittest.main()

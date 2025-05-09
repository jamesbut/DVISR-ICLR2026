# Functions for analysing results from experiments

import json
import matplotlib.pyplot as plt
from algorithms.vicatsr.q import q
from domains.domain_factory import create_domain
from algorithms.vicatsr.vicatsr import log_likelihood
from algorithms.vicatsr.equation import Equation
from util.norms import normalise_value
import numpy as np
import os


def analyse_results(run_dir):

    # Assume that a run directory is given
    exp_dir = '/'.join(run_dir.split('/')[:-1])

    # Read config
    with open(exp_dir + '/config.json', 'r') as file:
        config = json.load(file)

    # Read results
    with open(run_dir + '/results.json', 'r') as file:
        results = json.load(file)

    # print(json.dumps(results, indent=4))
    print('True z:', results['true_z'])
    print(f'Best z: {json.dumps(results["best_z"], indent=4)}')
    print('Epoch true model located', results['epoch_true_model_located'])
    print('ELBO max:', max(results['all_elbos']))

    # Read in best model
    best_z = Equation(infix_str=results['best_z']['eq'],
                      token_set=results['q']['token_set'])

    plot_results(results)

    # Create network paths to reflect the directory that the data is currently
    # in
    results['q']['net_path'] = os.getcwd() + '/' + run_dir + '/net.pt'
    results['init_q']['net_path'] = os.getcwd() + '/' + run_dir + '/init_net.pt'

    # Read q(z)
    q_z = q.from_json(results['q'])

    # Read initial q(z) if it exists
    init_q_z = q.from_json(results['init_q']) if 'init_q' in results else None

    # Create domain
    domain = create_domain(config['domain'])

    # Apply masks to best model
    best_z.apply_pre_softmax_mask(config['algorithm']['max_num_tokens'],
                                  q_z._net_masks)

    # Sample from q(z) and plot
    sample_and_plot(domain, q_z, init_q_z, best_z)


def plot_results(results):

    if results['kl_divs']:
        plt.plot(range(len(results['kl_divs'])), results['kl_divs'],
                 label='KL divergence')
        plt.legend()
        plt.show()

    if results['all_elbos']:
        elbos_len = len(results['all_elbos'])

        plt.plot(range(elbos_len), results['all_elbos'],
                 label='ELBO')

        if results['log_ev']:
            log_ev = results['log_ev']
            plt.plot(range(elbos_len), [log_ev] * elbos_len,
                     label=f'log p(x): {log_ev:.5f}')

        plt.legend()
        plt.show()

    if results['all_lls']:
        plt.plot(range(len(results['all_lls'])), results['all_lls'],
                 label='p(x|z)')
        plt.legend()
        plt.show()


def sample_and_plot(domain, q, init_q, best_z):

    data = domain.create_data()

    if data['x'].shape[1] > 1:
        print('WARNING: Cannot plot models when the number '
              'of independent variables is larger than 1')
        return

    models = []
    for i in range(10):
        model = q.sample()
        pdf = q.pdf(model)
        ll = log_likelihood(data, model)
        models.append((model, ll, pdf))

    # Sort models by log likelihoods so the plot is a little clearer
    models = sorted(models, key=lambda m: m[1], reverse=True)

    # Check whether all likelihoods are the same
    if all(z[1] == models[0][1] for z in models):
        opacities = [1.0] * len(models)
    else:
        # Vary opacities based upon relative log likelihood
        opacities = []
        max_ll = max(models, key=lambda x: x[1])[1]
        min_ll = min(models, key=lambda x: x[1])[1]
        for m in models:
            opacities.append(normalise_value(m[1], min_ll, max_ll,
                                             0.1, 0.9999999))

    # Create wide range of x values according to domain spec in order to
    # plot model smoothly
    x = domain.create_x(num_vals=1001)
    x = np.sort(x, axis=0)

    for m, o in zip(models, opacities):
        y = m[0].evaluate(x)
        if y is not None:
            plt.plot(x, y,
                     label=f'y = {m[0].get_infix()} | '
                           f'y = {m[0].get_infix(True)} '
                           f'(ln p(x|z): {m[1]:.2f}, q(z): {m[2]:.3f})',
                     c='tab:blue', alpha=o)

    # Sample from initial q is given
    init_models = []
    for i in range(10):
        model = init_q.sample()
        pdf = q.pdf(model)
        ll = log_likelihood(data, model)
        init_models.append((model, ll, pdf))

    for m in init_models:
        y = m[0].evaluate(x)
        if y is None:
            continue
        plt.plot(x, y, c='tab:orange', alpha=0.3, linestyle='--')

    # Plot best model
    plt.plot(x, best_z.evaluate(x), c='r')

    # Plot data points
    plt.scatter(data['x'][:, 0], data['y'], c='r', marker='x')

    # Report some metrics
    sampled_metrics = {
        'init_q_models': {
            'Mean log likelihood':
                (sum([m[1] for m in init_models]) / len(init_models))
        },
        'optimised_q_models': {
            'Mean log likelihood': sum([m[1] for m in models]) / len(models)
        }
    }
    print(json.dumps(sampled_metrics, indent=4))

    print('\nBest model:\n\n')
    print(f'y = {best_z.get_infix()}')
    print(f'y = {best_z.get_infix(True)}')
    # Check for invalid model
    y = best_z.evaluate(x)
    if y is None:
        print('INVALID')
    print(f'log p(x|z) = {log_likelihood(data, best_z)}')
    print(f'p(z) = {q.pdf(best_z)}\n')

    # Print models sampled from q(z)
    print('\nSampled models:\n\n')
    for m in models:

        print(f'y = {m[0].get_infix()}')
        print(f'y = {m[0].get_infix(True)}')

        # Check for invalid model
        y = m[0].evaluate(x)
        if y is None:
            print('INVALID')

        print(f'log p(x|z) = {m[1]}')
        print(f'p(z) = {m[2]}\n')

    # Print models sampled from the initial q(z)
    print('\nInitial sampled models:\n\n')
    for m in init_models:

        print(f'y = {m[0].get_infix()}')
        print(f'y = {m[0].get_infix(True)}')

        # Check for invalid model
        y = m[0].evaluate(x)
        if y is None:
            print('INVALID')

        print(f'log p(x|z) = {m[1]}')
        print(f'p(z) = {m[2]}\n')

    plt.legend()
    plt.show()

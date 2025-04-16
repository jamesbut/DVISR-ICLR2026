# Functions for analysing results from experiments

import json
import matplotlib.pyplot as plt
from algorithms.vicatsr.q import q


def analyse_results(exp_dir):

    # Read results
    with open(exp_dir + '/results.json', 'r') as file:
        results = json.load(file)

    print(json.dumps(results, indent=4))

    plot_results(results)

    # Read q(z)
    q_z = q.from_json(results['q'])


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

# Functions for analysing results from experiments

import json
import matplotlib.pyplot as plt
from algorithms.vicatsr.q import q
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm
from algorithms.vicatsr.vicatsr import log_likelihood, likelihood
from algorithms.vicatsr.equation import Equation
from algorithms.vicatsr.integrators import integrate_q_z_c
from util.norms import normalise_value
from util.permutations import compute_permutations
import numpy as np
import os
from pathlib import Path
from util.stats import median
import copy


def analyse_results(args):

    run_dir = args.analyse

    # Determine whether a run or an experiment directory has been provided
    parent_dir = run_dir.split('/')[-1]

    # Set exp directory accordingly
    if 'run_' in parent_dir:
        exp_dir = '/'.join(run_dir.split('/')[:-1])
    else:
        exp_dir = run_dir

    run_dirs = [run_dir]

    # Read config
    with open(exp_dir + '/config.json', 'r') as file:
        config = json.load(file)

    # Old file setup - one run without run_ directories
    if os.path.exists(exp_dir + '/results.json'):

        with open(exp_dir + '/results.json', 'r') as file:
            results = json.load(file)

        all_results = [results]

        plot_results(all_results, args.save)

    # Single run directory given
    elif exp_dir != run_dir:

        with open(run_dir + '/results.json', 'r') as file:
            results = json.load(file)

        all_results = [results]

        plot_results(all_results, args.save)

    # Multiple runs given in the form of an experiment directory
    else:

        # Read results from all runs if results.json has been created.
        # If not, the run has typically not finished so do not include.
        run_dirs = [p for p in Path(exp_dir).iterdir()
                    if p.is_dir() and os.path.exists(str(p) + '/results.json')]
        all_results = []
        for rd in run_dirs:
            with open(str(rd) + '/results.json', 'r') as file:
                all_results.append(json.load(file))

        # Plot all results
        plot_results(all_results, args.save)

        # Use run with the median final ELBO for analysis below
        final_elbos = [r['all_elbos'][-1] for r in all_results]
        med_elbo, med_idx = median(final_elbos, reverse_sort=True,
                                   prefer_lower=True)
        results = all_results[med_idx]
        run_dir = str(run_dirs[med_idx])

    # print(json.dumps(results, indent=4))
    print('True z:', results['true_z'])
    print(f'Best z: {json.dumps(results["best_z"], indent=4)}')
    print('Epoch true model located', results['epoch_true_model_located'])
    print('ELBO max:', max(results['all_elbos']))

    # Read in best model
    best_z = Equation(infix_str=results['best_z']['eq'],
                      token_set=results['q']['token_set'])

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

    # Create algorithm, data and initialise
    alg = create_algorithm(config['algorithm'], domain)
    data = domain.create_data()
    alg._initialise(data)

    # Apply masks to best model
    best_z.apply_pre_softmax_mask(config['algorithm']['max_num_tokens'],
                                  q_z._net_masks)

    # Sample from q(z) and plot
    # sample_and_plot(domain, q_z, init_q_z, best_z, alg, data)

    # Calculate true posteriors and compare to q(z)
    if args.true_pos:
        calc_true_posteriors(config, alg, data, all_results, run_dirs)

    # Plot distributions over c
    plot_c_distrs(alg, data, q_z)


def plot_results(results, save):

    # Create figures directory if it doesn't yet exist
    if save:
        os.makedirs('figures', exist_ok=True)

    # Collate results
    kl_divs = []
    elbos = []
    lls = []
    l_joints = []
    for r in results:
        if 'kl_divs' in r and r['kl_divs']:
            kl_divs.append(r['kl_divs'])
        if 'all_elbos' in r and r['all_elbos']:
            elbos.append(r['all_elbos'])
        if 'all_lls' in r and r['all_lls']:
            lls.append(r['all_lls'])
        if 'all_l_joints' in r and r['all_l_joints']:
            l_joints.append(r['all_l_joints'])

    kl_divs = np.array(kl_divs)
    elbos = np.array(elbos)
    lls = np.array(lls)
    l_joints = np.array(l_joints)

    x = range(elbos.shape[1])

    plot_kl_divs(x, kl_divs, save)
    plot_elbos(x, elbos, save, results)
    plot_log_likelihoods(x, lls, save)
    plot_log_joints(x, l_joints, save)


# Plot KL divergences
def plot_kl_divs(x, kl_divs, save):

    if kl_divs.size != 0:
        medians = np.median(kl_divs, axis=0)
        q1s = np.percentile(kl_divs, 25, axis=0)
        q3s = np.percentile(kl_divs, 75, axis=0)

        plt.plot(x, medians, label='KL divergence')

        plt.fill_between(x, q1s, q3s, color='lightblue', alpha=0.5)

        plt.plot(x, [0] * len(x), label='y = 0')

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.tight_layout()

        if save:
            plt.savefig('figures/kl_divs.svg', format='svg')

        plt.show()


# Plot ELBOs
def plot_elbos(x, elbos, save, results):

    if elbos.size != 0:
        medians = np.median(elbos, axis=0)
        q1s = np.percentile(elbos, 25, axis=0)
        q3s = np.percentile(elbos, 75, axis=0)

        plt.plot(x, medians, label='ELBO')

        plt.fill_between(x, q1s, q3s, color='lightblue', alpha=0.5)

        if results[0]['log_ev']:
            log_ev = results[0]['log_ev']
            plt.plot(x, [log_ev] * len(x),
                     label=f'log p(X,y): {log_ev:.5f}')

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')
        plt.tight_layout()

        if save:
            plt.savefig('figures/elbos.svg', format='svg')

        plt.show()


# Plot log likelihoods
def plot_log_likelihoods(x, lls, save):

    if lls.size != 0:
        medians = np.median(lls, axis=0)
        q1s = np.percentile(lls, 25, axis=0)
        q3s = np.percentile(lls, 75, axis=0)

        plt.plot(x, medians)

        plt.fill_between(x, q1s, q3s, color='lightblue', alpha=0.5)

        plt.xlabel('Epoch')
        plt.ylabel('log p(x|z)')
        plt.tight_layout()

        if save:
            plt.savefig('figures/log_likelihoods.svg', format='svg')

        plt.show()


# Plot log joints
def plot_log_joints(x, l_joints, save):

    if l_joints.size != 0:
        medians = np.median(l_joints, axis=0)
        q1s = np.percentile(l_joints, 25, axis=0)
        q3s = np.percentile(l_joints, 75, axis=0)

        plt.plot(x, medians)

        plt.fill_between(x, q1s, q3s, color='lightblue', alpha=0.5)

        plt.xlabel('Epoch')
        plt.ylabel('log p(x,z)')
        plt.tight_layout()

        if save:
            plt.savefig('figures/log_joints.svg', format='svg')

        plt.show()


# Sample from q(z) and plot
def sample_and_plot(domain, q, init_q, best_z, alg, data):

    if data['x'].shape[1] > 1:
        print('WARNING: Cannot plot models when the number '
              'of independent variables is larger than 1')
        return

    models = []
    for i in range(10):
        model = q.sample()
        pdf = q.pdf(model)
        ll = log_likelihood(data, model, alg._likelihood_sd,
                            alg._max_num_tokens, alg._net_masks)
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
        ll = log_likelihood(data, model, alg._likelihood_sd,
                            alg._max_num_tokens, alg._net_masks)
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
    ll = log_likelihood(data, best_z, alg._likelihood_sd,
                        alg._max_num_tokens, alg._net_masks)
    print(f'log p(x|z) = {ll}')
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


# Plot priors, likelihoods, joints and posterior for c values
def plot_c_distrs(alg, data, q):

    all_exps = alg._enumerate_expressions(data)

    for e in all_exps:

        if e.num_distr_consts() == 0:
            continue

        if e.num_distr_consts() > 1:
            print('Skipping: y =', e.get_infix())
            continue

        print('y =', e.get_infix())

        num_distr_consts = e.num_distr_consts()

        lbs = [-5.0] * num_distr_consts
        ubs = [5.0] * num_distr_consts
        step_size = 0.01
        step_sizes = [step_size] * num_distr_consts
        x = compute_permutations(lbs, ubs, step_sizes)

        exps = [copy.deepcopy(e) for _ in range(len(x))]
        for c, e in zip(x, exps):
            e.set_distr_consts(c)

        priors = [alg._prior(z) for z in exps]
        likelihoods = [likelihood(data, z, alg._likelihood_sd,
                                  alg._max_num_tokens, alg._net_masks)
                       for z in exps]
        joints = [l * p for p, l in zip(priors, likelihoods)]
        evidence = alg.evidence(data, all_exps)
        posteriors = [j / evidence for j in joints]
        qs = [q.pdf(z).item() for z in exps]

        # for c, l in zip(x, likelihoods):
        #     print(str(c) + '        ' + str(l))

        prior_max = x[np.argmax(priors)]
        likelihood_max = x[np.argmax(likelihoods)]
        joint_max = x[np.argmax(joints)]
        posterior_max = x[np.argmax(posteriors)]
        q_max = x[np.argmax(qs)]

        print('Evidence:', evidence)
        print('Prior max:', prior_max)
        print('Likelihood max:', likelihood_max)
        print('Joint max:', joint_max)
        print('Posterior max:', posterior_max)
        print('q max:', q_max)
        print('----------------------')

        plt.plot(x, priors, label='Prior')
        plt.plot(x, likelihoods, label='Likelihood')
        plt.plot(x, joints, label='Joint')
        plt.plot(x, posteriors, label='Posterior')
        plt.plot(x, qs, label='q(z)')

        plt.legend()

        plt.show()


# Enumerate models and calculate true posteriors
def calc_true_posteriors(config, alg, data, all_results, run_dirs):

    true_posteriors, all_exps = alg.posteriors(data)

    q_z_vals = []

    # Calculate sample mean of q_z and standard deviation for quantification
    # of uncertainty by considering all runs
    for r, rd in zip(all_results, run_dirs):

        # Create network paths to reflect the directory that the data is
        # currently in
        r['q']['net_path'] = os.getcwd() + '/' + str(rd) + '/net.pt'
        r['init_q']['net_path'] = os.getcwd() + '/' + str(rd) + '/init_net.pt'

        # Read q(z)
        q_z = q.from_json(r['q'])
        alg._r = q_z

        # kl_divergence = alg.kl_divergence(data, num_samples=1000)
        # print('KL divergence:', kl_divergence)

        if alg._posterior_integration:
            q_z_vals.append(integrate_q_z_c(q_z, all_exps))

        # Otherwise just calculate q(z,c) for whatever values are
        # currently set to c
        else:
            q_z_vals.append([q_z.pdf(z).item() for z in all_exps])

    q_z_vals = np.array(q_z_vals)
    q_z_means = np.mean(q_z_vals, axis=0)
    q_z_stds = np.std(q_z_vals, axis=0)
    q_z_medians = np.median(q_z_vals, axis=0)
    q_z_q1s = np.percentile(q_z_vals, 25, axis=0)
    q_z_q3s = np.percentile(q_z_vals, 75, axis=0)
    q_z_iqrs = q_z_q3s - q_z_q1s

    # Order all models by q(z) and print
    all_z = [(z, p_z_x, q_z_med, q_z_q1, q_z_q3, q_z_iqr,
              log_likelihood(data, z, alg._max_num_tokens, alg._net_masks),
              q_z.get_consts_params(z))
             for z, p_z_x, q_z_med, q_z_q1, q_z_q3, q_z_iqr in
             zip(all_exps, true_posteriors, q_z_medians,
                 q_z_q1s, q_z_q3s, q_z_iqrs)]
    all_z = sorted(all_z, key=lambda z: z[2], reverse=True)

    # Get longest eq string in order to format nicely
    eq_str_length = max(len(z[0].get_infix()) for z in all_z)

    for i, z in enumerate(all_z):
        out_str = (f'z: {z[0].get_infix():<{eq_str_length+3}} '
                   f'z: {z[0].get_infix(simplify=True):<25} '
                   f'p(z|x): {z[1]:.10f} '
                   f'q(z): {z[2]:.10f} [{z[3]:.10f}, {z[4]:.10f}] '
                   f'p(x|z): {z[6]:.10f}')
        if alg._distr_over_consts:
            out_str += f'   q consts params: {z[7]}'
        print(out_str)
        if i > 100:
            break

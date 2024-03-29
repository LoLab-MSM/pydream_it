import sys
import importlib
import os
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


def is_numbers(inputstring):
    return all(char.isdigit() for char in inputstring)


def parse_directive(directive, priors, no_sample):
    words = directive.split()
    if words[1] == 'prior':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]
        priors[par] = (words[3], float(words[4]))
    elif words[1] == 'no-sample':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]
        no_sample.append(par)
    return


def prune_no_samples(parameters, no_sample):
    pruned_pars = [p for p in parameters if p.name not in no_sample]
    return pruned_pars


def write_norm_param(p_name, p_val, p_scale):
    line = "sampled_params_list.append(SampledParam(norm, loc=np.log10({}), scale={}))  # {}\n".\
        format(p_val, p_scale, p_name)
    return line


def write_uniform_param(p_name, p_val, p_scale):
    line = "sampled_params_list.append(SampledParam(uniform, loc=np.log10({})-{}, scale={}))  # {}\n".\
        format(p_val, 0.5*p_scale, p_scale, p_name)
    return line


def plot_param_dist(samples, labels, **kwargs):
    # error check
    if len(samples[0]) != len(labels):
        print("Error: 'ndims' (%d) and 'labels' (%d) are not the same length. Please try again." %
              (len(samples[0]), len(labels)))
        quit()
    ndims = len(labels)
    # set plot parameters
    fscale = np.ceil(ndims / 16)
    figsize = kwargs.get('figsize', fscale * np.array([6.4, 4.8]))
    labelsize = kwargs.get('labelsize', 10 * max(1, (2/5 * fscale)))
    fontsize = kwargs.get('fontsize', 10 * max(1, (3/5 * fscale)))
    ncols = kwargs.get('ncols', int(np.ceil(np.sqrt(ndims))))
    nrows = int(np.ceil(ndims/ncols))
    # create figure
    colors = sns.color_palette(n_colors=ndims)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, constrained_layout=True,
                            figsize=figsize)
    row = 0
    col = 0
    for dim in range(ndims):
        print(dim, end=' ')
        sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True, ax=axs[row][col])
        axs[row][col].set_yticklabels([])
        axs[row][col].set_ylabel(None)
        axs[row][col].set_title(labels[dim], fontsize=labelsize)
        axs[row][col].tick_params(axis='x', labelsize=labelsize)
        col += 1
        if col % ncols == 0:
            col = 0
            row += 1
    print()
    fig.supxlabel(r'log$_{10}$ value', fontsize=fontsize)
    fig.supylabel('Density', fontsize=fontsize)
    # delete extra plots
    if col > 0:
        while col < ncols:
            fig.delaxes(axs[row][col])
            col += 1
    # save plots
    suffix = kwargs.get('suffix', '')
    plt.savefig('fig_PyDREAM_histograms' + suffix)


def plot_log_likelihood(log_ps, cutoff=None):
    plt.figure()
    nchains = len(log_ps)
    burnin = int(len(log_ps[0]) / 2)  # calculate mean and variance for last half of steps
    log_ps_max = -np.inf
    log_ps_mean = 0
    log_ps_var = 0
    for chain in range(nchains):
        plt.plot(range(len(log_ps[chain])), log_ps[chain], label='chain %d' % chain)
        log_ps_max = np.max(log_ps[chain]) if log_ps_max < np.max(log_ps[chain]) else log_ps_max
        log_ps_mean += np.mean(log_ps[chain][burnin:]) / nchains
        log_ps_var += np.var(log_ps[chain][burnin:]) / nchains  # this is the mean of the variances, but that's fine
    top = np.ceil(log_ps_mean + 5 * np.sqrt(log_ps_var))
    bottom = np.floor(log_ps_mean - 20 * np.sqrt(log_ps_var))
    print('max: %g, mean: %g, sdev: %g, top: %g, bottom: %g' %
          (log_ps_max, log_ps_mean, np.sqrt(log_ps_var), top, bottom))
    if cutoff is not None:
        plt.axhline(log_ps_mean - cutoff * np.sqrt(log_ps_var), color='k', ls='--', lw=2)
    plt.ylim(bottom=bottom, top=top)
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('fig_PyDREAM_log_ps')


def plot_time_courses(observables, sim_tspan, sim_output, counts=None, exp_data=None, fill_between=(5, 95), **kwargs):
    plt.figure()
    output = np.copy(sim_output)
    # remove any simulations that produced NaNs
    idx_remove = [i for i in range(len(output)) if np.any(np.isnan(output[i][observables[0]]))]
    if len(idx_remove) > 0:
        output = np.delete(output, idx_remove, axis=0)
        if counts is not None:
            counts = np.delete(counts, idx_remove, axis=0)
    # if applicable, use 'counts' to generate full set of simulation outputs for correct weighting for plots
    if counts is not None:
        output = np.repeat(output, counts, axis=0)
    # make plots
    for obs_name in observables:
        # plot simulated data as a percent envelope
        yvals = np.array([out[obs_name] for out in output])
        yvals_min = np.percentile(yvals, fill_between[0], axis=0)
        yvals_max = np.percentile(yvals, fill_between[1], axis=0)
        plt.fill_between(sim_tspan, yvals_min, yvals_max, alpha=0.5)
        # plot experimental data
        if exp_data is not None:
            exp_time = exp_data[0]
            exp_mean = exp_data[1]
            exp_sdev = exp_data[2]
            plt.errorbar(exp_time[obs_name], exp_mean[obs_name], yerr=exp_sdev[obs_name],
                         capsize=6, fmt='o', ms=8, label=obs_name)
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(loc=0)
    plt.tight_layout()
    suffix = kwargs.get('suffix', '')
    plt.savefig('fig_PyDREAM_time_courses' + suffix)


def get_unique_samples_for_simulation(samples, log_ps, cutoff=None):
    # error check
    if len(samples) != len(log_ps):
        print("Error: 'samples' (%d) and 'log_ps' (%d) are not the same length. Please try again." %
              (len(samples), len(log_ps)))
        quit()
    # only run simulations for unique parameter sets
    samples, idx_unique, counts = np.unique(samples, return_index=True, return_counts=True, axis=0)
    # prune parameter sets based on log_ps
    if cutoff is not None:
        avg = np.mean(log_ps)
        sdev = np.sqrt(np.var(log_ps))
        print('log_ps: avg = %g, sdev = %g, avg-%d*sdev = %g' % (avg, sdev, cutoff, avg-cutoff*sdev))
        # remove parameter sets that have a log_p less than 'cutoff' sdevs below the mean
        idx_remove = [i for i in range(len(log_ps[idx_unique])) if log_ps[idx_unique][i] < (avg - cutoff * sdev)]
        if len(idx_remove) > 0:
            samples = np.delete(samples, idx_remove, axis=0)
            counts = np.delete(counts, idx_remove, axis=0)
    return samples, counts


if __name__ == '__main__':

    # User settings
    include_init_params = True
    default_prior_shape = ('norm', 2.0)
    use_GR_converge = True
    try_plot = True

    # Read input arguments
    model_file = sys.argv[1]

    model_path = os.path.split(model_file)[0]
    model_file = os.path.split(model_file)[1]
    exp_data_avg = "%s_exp_data_avg_0.csv" % model_file[:-3]
    exp_data_se = "%s_exp_data_se_0.csv" % model_file[:-3]
    exp_data_time = "%s_exp_data_time_0.csv" % model_file[:-3]

    print("Using model from file: {}".format(model_file))
    print("The default prior shape is: {}".format(default_prior_shape))
    model_module_name = '.'.join([model_path.replace("/", "."), model_file[:-3]])
    model_module = importlib.import_module(model_module_name)
    model = getattr(model_module, 'model')
    priors = dict()
    no_sample = list()

    # Read the file and parse any #PYDREAM_IT directives
    print("Parsing the model for any #PYDREAM_IT directives...")
    with open(os.path.join(model_path, model_file), 'r') as file_obj:
        for line in file_obj:
            words = line.split()
            if len(words) > 1:
                if words[0] == '#PYDREAM_IT':
                    parse_directive(line, priors, no_sample)

    # now we need to extract a list of kinetic parameters
    print("Inspecting the model and pulling out kinetic parameters...")
    parameters = model.parameters if include_init_params else model.parameters_rules()
    # remove parameters flagged with 'no-sample' in the model file
    parameters = prune_no_samples(parameters, no_sample)
    print("Found the following parameters:")
    print("{}".format(parameters))
    # default the priors to normal distributions (norm) with sdev = 2
    for p in parameters:
        name = p.name
        if name not in priors.keys():
            priors[name] = default_prior_shape

    # Obtain mask of sampled parameters to run simulation in the likelihood function
    parameters_idxs = [model.parameters.index(p) for p in parameters]
    rates_mask = [i in parameters_idxs for i in range(len(model.parameters))]
    param_values = [p.value for p in model.parameters]

    out_file = open(os.path.join(model_path, "calibrate_pydream_" + model_file), 'w')

    print("Writing to PyDREAM run script: run_pydream_{}".format(model_file))
    out_file.write("\"\"\"\nGenerated by pydream_it\n")
    out_file.write("PyDREAM run script for {} \n".format(model_file))
    out_file.write("\"\"\"")
    out_file.write("\n")
    out_file.write("from pydream.core import run_dream\n")
    out_file.write("from pysb.simulator import ScipyOdeSimulator\n")
    out_file.write("import numpy as np\n")
    out_file.write("from pydream.parameters import SampledParam\n")
    out_file.write("from pydream.convergence import Gelman_Rubin\n")
    out_file.write("from scipy.stats import norm, uniform\n")
    out_file.write("from %s import model\n" % model_file[:-3])
    out_file.write("import os\n")
    out_file.write("import re\n")
    out_file.write("\n")
    out_file.write("# DREAM Settings\n")
    out_file.write("# Number of chains - should be at least 3.\n")
    out_file.write("nchains = 5\n")
    out_file.write("# Number of iterations\n")
    out_file.write("niterations = 50000\n")
    out_file.write("\n")
    out_file.write("# Initialize PySB solver object for running simulations. Simulation timespan should match " +
                   "experimental data.\n")
    out_file.write("files = sorted(os.listdir('.'))\n")
    out_file.write("exp_time_files = [f for f in files if re.search(r'exp_data_time_\\d+', f)]\n")
    out_file.write("experiments_time = [np.genfromtxt(file, delimiter=',', names=True) for file in exp_time_files]\n")
    out_file.write("n_experiments = len(experiments_time)\n")
    out_file.write("tspan = []\n")
    out_file.write("tspan_mask = []\n")
    out_file.write("for exp_time in experiments_time:\n")
    out_file.write("    tspan.append([])\n")
    out_file.write("    for name in exp_time.dtype.names:\n")
    out_file.write("        tspan[-1] += [t for t in exp_time[name] if not np.isnan(t)]\n")
    out_file.write("    tspan[-1] = sorted(list(set(tspan[-1])))  # get a common set of time points for simulations\n")
    out_file.write("    tspan_mask.append({})  # for each species, need to mark which time points we have data for\n")
    out_file.write("    for name in exp_time.dtype.names:\n")
    out_file.write("        tspan_mask[-1][name] = [False] * len(tspan[-1])\n")
    out_file.write("        for i in range(len(tspan[-1])):\n")
    out_file.write("            if tspan[-1][i] in exp_time[name]:\n")
    out_file.write("                tspan_mask[-1][name][i] = True\n")
    out_file.write("solver = ScipyOdeSimulator(model)\n")
    out_file.write("parameters_idxs = " + str(parameters_idxs)+"\n")
    out_file.write("rates_mask = " + str(rates_mask)+"\n")
    out_file.write("param_values = np.array([p.value for p in model.parameters])\n")
    out_file.write("\n")
    out_file.write("# USER must add commands to import/load any experimental data for use in the likelihood " +
                   "function!\n")
    out_file.write("exp_avg_files = [f for f in files if re.search(r'exp_data_avg_\\d+', f)]\n")
    out_file.write("experiments_avg = [np.genfromtxt(file, delimiter=',', names=True) for file in exp_avg_files]\n")
    out_file.write("exp_se_files = [f for f in files if re.search(r'exp_data_se_\\d+', f)]\n")
    out_file.write("experiments_se = [np.genfromtxt(file, delimiter=',', names=True) for file in exp_se_files]\n")
    out_file.write("like_data = []\n")
    out_file.write("for exp_avg, exp_se in zip(experiments_avg, experiments_se):\n")
    out_file.write("    like_data.append({})\n")
    out_file.write("    for name in exp_avg.dtype.names:\n")
    out_file.write("        # remove any nans, which will happen if the time points are different for different " +
                   "species\n")
    out_file.write("        avg = [e for e in exp_avg[name] if not np.isnan(e)]\n")
    out_file.write("        se = [e for e in exp_se[name] if not np.isnan(e)]\n")
    out_file.write("        like_data[-1][name] = norm(loc=avg, scale=se)\n")
    out_file.write("\n\n")
    out_file.write("# USER must define a likelihood function!\n")
    out_file.write("def likelihood(position):\n")
    out_file.write("    y = np.copy(position)\n")
    out_file.write("    logp_data = [0] * n_experiments\n")
    out_file.write("    for n in range(n_experiments):\n")
    out_file.write("        param_values[rates_mask] = 10 ** y\n")
    out_file.write("        sim = solver.run(tspan=tspan[n], param_values=param_values).all\n")
    out_file.write("        for sp in like_data[n].keys():\n")
    out_file.write("            logp_data[n] += np.sum(like_data[n][sp].logpdf(sim[sp][tspan_mask[n][sp]]))\n")
    out_file.write("        if np.isnan(logp_data[n]):\n")
    out_file.write("            logp_data[n] = -np.inf\n")
    out_file.write("    return sum(logp_data)\n")
    out_file.write("\n\n")

    # write the sampled params lines
    out_file.write("sampled_params_list = list()\n")
    for p in parameters:
        name = p.name
        value = p.value
        prior_shape = priors[name][0]
        scale = priors[name][1]
        print("Will sample parameter {} with {} prior around log10({}) = {}".format(name, priors[name], value,
                                                                                    np.log10(value)))
        if prior_shape == 'uniform':
            line = write_uniform_param(name, value, scale)
        else:
            line = write_norm_param(name, value, scale)
        out_file.write(line)

    # write the main part of the script
    out_file.write("\n")
    out_file.write("if __name__ == '__main__':\n\n")

    out_file.write("    sampled_params, log_ps = run_dream(parameters=sampled_params_list,\n")
    out_file.write("                                       likelihood=likelihood,\n")
    out_file.write("                                       niterations=niterations,\n")
    out_file.write("                                       nchains=nchains,\n")
    out_file.write("                                       multitry=False,\n")
    out_file.write("                                       gamma_levels=4,\n")
    out_file.write("                                       adapt_gamma=True,\n")
    out_file.write("                                       history_thin=1,\n")
    out_file.write("                                       model_name=\'dreamzs_%dchain\' % nchains,\n")
    out_file.write("                                       verbose=True)\n")
    out_file.write("    total_iterations = niterations\n")
    out_file.write("    burnin = int(total_iterations / 2)\n")
    out_file.write("    # Save sampling output (sampled parameter values and their corresponding logps).\n")
    out_file.write("    for chain in range(len(sampled_params)):\n")
    out_file.write("        np.save(\'dreamzs_%dchain_sampled_params_chain_%d_%d\' %\n" +
                   "                (nchains, chain, total_iterations), sampled_params[chain])\n")
    out_file.write("        np.save(\'dreamzs_%dchain_logps_chain_%d_%d\' % (nchains, chain, total_iterations), " +
                   "log_ps[chain])\n")
    out_file.write("    old_samples = sampled_params\n\n")

    if use_GR_converge:
        out_file.write("    # Check convergence and continue sampling if not converged\n")
        out_file.write("    GR = Gelman_Rubin(sampled_params)\n")
        out_file.write("    print('At iteration: ', total_iterations, ' GR = ', GR)\n")
        out_file.write("    np.savetxt(\'dreamzs_%dchain_GelmanRubin_iteration_%d.txt\' % " +
                       "(nchains, total_iterations), GR)\n")
        out_file.write("    if np.any(GR > 1.2):\n")
        out_file.write("        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]\n")
        out_file.write("        converged = False\n")
        out_file.write("        while not converged:\n")
        out_file.write("            total_iterations += niterations\n")
        out_file.write("            burnin += niterations\n")
        out_file.write("            sampled_params, log_ps = run_dream(parameters=sampled_params_list,\n")
        out_file.write("                                               likelihood=likelihood,\n")
        out_file.write("                                               niterations=niterations,\n")
        out_file.write("                                               nchains=nchains,\n")
        out_file.write("                                               start=starts,\n")
        out_file.write("                                               multitry=False,\n")
        out_file.write("                                               gamma_levels=4,\n")
        out_file.write("                                               adapt_gamma=True,\n")
        out_file.write("                                               history_thin=1,\n")
        out_file.write("                                               model_name=\'dreamzs_%dchain\' % nchains,\n")
        out_file.write("                                               verbose=True,\n")
        out_file.write("                                               restart=True)\n")

        # Save sampling output (sampled parameter values and their corresponding logps)
        out_file.write("            for chain in range(len(sampled_params)):\n")
        out_file.write("                np.save('dreamzs_%dchain_sampled_params_chain_%d_%d' %\n" +
                       "                        (nchains, chain, total_iterations), sampled_params[chain])\n")
        out_file.write("                np.save('dreamzs_%dchain_logps_chain_%d_%d' % " +
                       "(nchains, chain, total_iterations), log_ps[chain])\n")

        out_file.write("            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) " +
                       "for chain in range(nchains)]\n")
        out_file.write("            GR = Gelman_Rubin(old_samples)\n")
        out_file.write("            print(\'At iteration: \', total_iterations, \' GR = \', GR)\n")
        out_file.write("            np.savetxt(\'dreamzs_%dchain_GelmanRubin_iteration_%d.txt\'" +
                       " % (nchains, total_iterations), GR)\n")

        out_file.write("            if np.all(GR < 1.2):\n")
        out_file.write("                converged = True\n\n")

    if try_plot:
        out_file.write("    # Plot output\n")
        out_file.write("    try:\n")
        out_file.write("        from pydream_it import plot_param_dist, plot_log_likelihood, plot_time_courses, \\\n")
        out_file.write("            get_unique_samples_for_simulation\n")
        out_file.write("\n")
        out_file.write("        total_iterations = len(old_samples[0])\n")
        # TODO: Can probably add code here to get samples for all n experiments to be used below (see TODOs)
        out_file.write("        # parameter distributions\n")
        out_file.write("        print('Plotting parameter distributions')\n")
        out_file.write("        samples = np.concatenate(tuple([old_samples[chain][burnin:, :] " +
                       "for chain in range(nchains)]))\n")
        out_file.write("        for n in range(n_experiments):\n")
        out_file.write("            samples_n = samples  # TODO: Add code to get samples for the nth experiment\n")
        out_file.write("            plot_param_dist(samples_n, [model.parameters[i].name for i in parameters_idxs],\n")
        out_file.write("                            suffix='_exp_%d' % n)\n")
        out_file.write("        # log likelihood\n")
        out_file.write("        print('Plotting log-likelihoods')\n")
        out_file.write("        log_ps = []\n")
        out_file.write("        n_files = int(total_iterations / niterations)\n")
        out_file.write("        for chain in range(nchains):\n")
        out_file.write("            log_ps.append(np.concatenate(\n")
        out_file.write("                tuple(np.load('dreamzs_%dchain_logps_chain_%d_%d.npy' % " +
                       "(nchains, chain, niterations * (i+1))).flatten()\n")
        out_file.write("                      for i in range(n_files))))\n")
        out_file.write("        plot_log_likelihood(log_ps, cutoff=2)\n")
        out_file.write("        # time courses\n")
        out_file.write("        print('Plotting time courses')\n")
        out_file.write("        log_ps = np.concatenate(tuple(log_ps[i][burnin:] for i in range(nchains)))\n")
        out_file.write("        for n in range(n_experiments):\n")
        out_file.write("            print('Experiment %d' % n)\n")
        out_file.write("            tspan = np.linspace(tspan[n][0], tspan[n][-1], " +
                       "int((tspan[n][-1] - tspan[n][0]) * 10 + 1))\n")
        out_file.write("            samples_n = samples  # TODO: Add code to get samples for the nth experiment\n")
        out_file.write("            samples_n, counts = get_unique_samples_for_simulation(samples_n, log_ps, " +
                       "cutoff=2)\n")
        out_file.write("            param_values = np.array([param_values] * len(samples_n))\n")
        out_file.write("            for i in range(len(param_values)):\n")
        out_file.write("                param_values[i][parameters_idxs] = 10 ** samples_n[i]\n")
        out_file.write("            print('Running %d simulations' % len(param_values))\n")
        out_file.write("            output_all = solver.run(tspan=tspan, param_values=param_values).all\n")
        out_file.write("            plot_time_courses(experiments_avg[n].dtype.names, tspan, output_all, " +
                       "counts=counts,\n")
        out_file.write("                              exp_data=(experiments_time[n], experiments_avg[n], " +
                       "experiments_se[n]),\n")
        out_file.write("                              suffix='_exp_%d' % n)\n")
        out_file.write("        print('DONE')\n")
        out_file.write("\n")
        out_file.write("    except ImportError:\n")
        out_file.write("        pass\n")

    out_file.write("\nelse:\n")
    out_file.write("    run_kwargs = {" +
                   "'parameters': sampled_params_list, 'likelihood': likelihood, 'niterations': niterations,\n " +
                   "                 'nchains': nchains, 'multitry': False, 'gamma_levels': 4, " +
                   "'adapt_gamma': True, 'history_thin': 1,\n " +
                   "                 'model_name': 'dreamzs_%dchain' % nchains, 'verbose': True}\n")

    out_file.close()
    print("pydream_it is complete!")
    print("END OF LINE.")

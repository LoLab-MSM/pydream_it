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
    line = "sp_{} = SampledParam(norm, loc=np.log10({}), scale={})\n".format(p_name, p_val, p_scale)
    return line


def write_uniform_param(p_name, p_val, p_scale):
    line = "sp_{} = SampledParam(uniform, loc=np.log10({})-{}, scale={})\n".format(p_name, p_val, 0.5*p_scale, p_scale)
    return line


def plot_param_dist(samples, labels):
    ndims = len(samples[0])
    colors = sns.color_palette(n_colors=ndims)
    for dim in range(ndims):
        plt.figure()
        sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
        plt.xlabel(r'log$_{10}$ %s' % labels[dim])
        plt.savefig('fig_PyDREAM_dimension_%d' % dim)


def plot_log_likelihood(log_ps):
    plt.figure()
    nchains = len(log_ps)
    maximums = []
    for chain in range(nchains):
        maximums.append(log_ps[chain].max())
        plt.plot(range(len(log_ps[chain])), log_ps[chain], label='chain %d' % chain)
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')
    top = np.ceil(np.max(maximums))
    bottom = 0.9 * top if top > 0 else 1.1 * top
    plt.ylim(bottom=bottom, top=top)
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('fig_PyDREAM_log_ps')


def plot_time_courses(sim_output, sim_tspan, exp_time, exp_mean, exp_sdev):
    plt.figure()
    for obs_name in exp_mean.dtype.names:
        # plot simulated data as a 95% envelope
        yvals = np.array([out[obs_name] for out in sim_output.all])
        yvals_5 = np.percentile(yvals, 5, axis=0)
        yvals_95 = np.percentile(yvals, 95, axis=0)
        plt.fill_between(sim_tspan, yvals_5, yvals_95, alpha=0.5)  # , color='0.75')
        # plot experimental data
        plt.errorbar(exp_time, exp_mean[obs_name], yerr=exp_sdev[obs_name], capsize=6, fmt='o', ms=8, label=obs_name)
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('fig_PyDREAM_time_courses')


if __name__ == '__main__':

    # Read input arguments
    model_file = sys.argv[1]

    model_path = os.path.split(model_file)[0]
    model_file = os.path.split(model_file)[1]
    exp_data_avg = "%s_exp_data_avg.csv" % model_file[:-3]
    exp_data_sd = "%s_exp_data_sd.csv" % model_file[:-3]
    exp_data_time = "%s_exp_data_time.csv" % model_file[:-3]

    print("Using model from file: {}".format(model_file))
    default_prior_shape = ('norm', 2.0)
    print("The default prior shape is: {}".format(default_prior_shape))
    use_GR_converge = True
    try_plot = True
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
    parameters = prune_no_samples(model.parameters_rules(), no_sample)
    print("Found the following kinetic parameters:")
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
    out_file.write("from " + model_module_name + " import model\n")
    out_file.write("\n")
    out_file.write("# DREAM Settings\n")
    out_file.write("# Number of chains - should be at least 3.\n")
    out_file.write("nchains = 5\n")
    out_file.write("# Number of iterations\n")
    out_file.write("niterations = 50000\n")
    out_file.write("\n")
    out_file.write("# Initialize PySB solver object for running simulations.  " +
                   "Simulation timespan should match experimental data.\n")
    out_file.write("experiments_time = np.genfromtxt(\"%s\", names=True)['time']\n" % exp_data_time)
    out_file.write("solver = ScipyOdeSimulator(model, tspan=experiments_time)\n")
    out_file.write("parameters_idxs = " + str(parameters_idxs)+"\n")
    out_file.write("rates_mask = " + str(rates_mask)+"\n")
    out_file.write("param_values = np.array([p.value for p in model.parameters])\n")
    out_file.write("\n")
    out_file.write("# USER must add commands to import/load any experimental data for use in the likelihood " +
                   "function!\n")
    out_file.write("experiments_avg = np.genfromtxt(\"%s\", delimiter=',', names=True)\n" % exp_data_avg)
    out_file.write("experiments_sd = np.genfromtxt(\"%s\", delimiter=',', names=True)\n" % exp_data_sd)
    out_file.write("like_data = {}\n")
    out_file.write("for sp in experiments_avg.dtype.names:\n")
    out_file.write("    like_data[sp] = norm(loc=experiments_avg[sp], scale=experiments_sd[sp])\n")
    out_file.write("\n\n")
    out_file.write("# USER must define a likelihood function!\n")
    out_file.write("def likelihood(position):\n")
    out_file.write("    y = np.copy(position)\n")
    out_file.write("    param_values[rates_mask] = 10 ** y\n")
    out_file.write("    sim = solver.run(param_values=param_values).all\n")
    out_file.write("    logp_data = 0\n")
    out_file.write("    for sp in like_data.keys():\n")
    out_file.write("        logp_data += np.sum(like_data[sp].logpdf(sim[sp]))\n")
    out_file.write("    if np.isnan(logp_data):\n")
    out_file.write("        logp_data = -np.inf\n")
    out_file.write("    return logp_data\n")
    out_file.write("\n\n")

    # write the sampled params lines
    out_file.write("sampled_params_list = list()\n")
    for p in parameters:
        name = p.name
        value = p.value
        prior_shape = priors[name][0]
        scale = priors[name][1]
        print("Will sample parameter {} with {} prior around {}".format(name, priors[name], value))
        if prior_shape == 'uniform':
            line = write_uniform_param(name, value, scale)
        else:
            line = write_norm_param(name, value, scale)
        ps_name = line.split()[0]
        out_file.write(line)
        out_file.write("sampled_params_list.append({})\n".format(ps_name))

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
        out_file.write("            sampled_params, log_ps = run_dream(parameters=sampled_params_list,\n")
        out_file.write("                                               likelihood=likelihood,\n")
        out_file.write("                                               niterations=niterations,\n")
        out_file.write("                                               nchains=nchains,\n")
        out_file.write("                                               start=starts,\n")
        out_file.write("                                               multitry=True,\n")
        out_file.write("                                               gamma_levels=4,\n")
        out_file.write("                                               adapt_gamma=True,\n")
        out_file.write("                                               history_thin=1,\n")
        out_file.write("                                               model_name=\'dreamzs_%dchain\' % nchains,\n")
        out_file.write("                                               verbose=False,\n")
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
        # Plot output
        out_file.write("    try:\n")
        out_file.write("        from pydream_it import plot_param_dist, plot_log_likelihood, plot_time_courses\n")
        out_file.write("\n")
        out_file.write("        total_iterations = len(old_samples[0])\n")
        out_file.write("        burnin = int(total_iterations / 2)\n")
        out_file.write("        # parameter distributions\n")
        out_file.write("        print('Plotting parameter distributions')\n")
        out_file.write("        samples = np.concatenate(tuple([old_samples[i][burnin:, :] " +
                       "for i in range(nchains)]))\n")
        out_file.write("        plot_param_dist(samples, [model.parameters[i].name for i in parameters_idxs])\n")
        out_file.write("        # log likelihood\n")
        out_file.write("        print('Plotting log-likelihoods')\n")
        out_file.write("        log_ps = []\n")
        out_file.write("        for chain in range(nchains):\n")
        out_file.write("            log_ps.append(np.load('dreamzs_%dchain_logps_chain_%d_%d.npy' % " +
                       "(nchains, chain, total_iterations)))\n")
        out_file.write("        plot_log_likelihood(log_ps)\n")
        out_file.write("        # time courses\n")
        out_file.write("        print('Plotting time courses')\n")
        out_file.write("        tspan = np.linspace(experiments_time[0], experiments_time[-1], " +
                       "len(experiments_time) * 10 + 1)\n")
        out_file.write("        param_values = np.array([param_values] * len(samples))\n")
        out_file.write("        for i in range(len(param_values)):\n")
        out_file.write("            param_values[i][parameters_idxs] = 10 ** samples[i]\n")
        out_file.write("        output = solver.run(tspan=tspan, param_values=param_values)\n")
        out_file.write("        plot_time_courses(output, tspan, experiments_time, experiments_avg, experiments_sd)\n")
        out_file.write("\n")

        out_file.write("    except ImportError:\n")
        out_file.write("        pass\n")

    out_file.write("\nelse:\n")
    out_file.write("    run_kwargs = {'parameters': sampled_params_list, 'likelihood': likelihood, " +
                   "'niterations': niterations,\n " +
                   "                 'nchains': nchains, 'multitry': False, 'gamma_levels': 4, " +
                   "'adapt_gamma': True, 'history_thin': 1,\n " +
                   "                 'model_name': 'dreamzs_%dchain' % nchains, 'verbose': True}\n")

    out_file.close()
    print("pydream_it is complete!")
    print("END OF LINE.")

import sys
import importlib

def is_numbers(inputString):
    return all(char.isdigit() for char in inputString)

def parse_directive(directive, priors, no_sample):
    words = directive.split()
    if words[1] == 'prior':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]    
        priors[par] = words[3]
    elif words[1] == 'no-sample':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]
        no_sample.append(par)
    return

def prune_no_samples(parameters, no_sample):
    pruned_pars = [parameter for parameter in parameters if parameter[0].name not in no_sample]

    return pruned_pars

def write_norm_param(p_name, p_val):
    line = "sp_{} = SampledParam(norm, loc=np.log10({}), scale=2.0)\n".format(p_name, p_val)
    return line

def write_uniform_param(p_name, p_val):
    line = "sp_{} = SampledParam(uniform, loc=np.log10({})-1.0, scale=2.0)\n".format(p_name, p_val)
    return line

model_file = sys.argv[1]
print("Using model from file: {}".format(model_file))
default_prior_shape = 'norm'
print("The default prior shape is: {}".format(default_prior_shape))
use_GR_converge = True
try_plot = True
#print(model_file)
model_module_name = model_file[:-3]
#print(model_module_name)
model_module = importlib.import_module(model_module_name)
model = getattr(model_module, 'model')
#print(model)
priors = dict()
no_sample = list()
#Read the file and parse any #PYDREAM_IT directives
print("Parsing the model for any #PYDREAM_IT directives...")
with open(model_file, 'r') as file_obj:
    for line in file_obj:
        words = line.split()
        if len(words) > 1:
            if words[0] == '#PYDREAM_IT':
                parse_directive(line, priors, no_sample)

#now we need to extract a list of kinetic parameters
parameters = list()
print("Inspecting the model and pulling out kinetic parameters...")
for rule in model.rules:
    #print(rule_keys)
    if rule.rate_forward:
        param = rule.rate_forward
        #print(param)
        parameters.append([param,'f'])
    if rule.rate_reverse:
        param = rule.rate_reverse
        #print(param)
        parameters.append([param, 'r'])
#print(parameters)
#print(no_sample)
parameters = prune_no_samples(parameters, no_sample)
#print(parameters)
print("Found the following kinetic parameters:")
print("{}".format(parameters))
#default the priors to norm - i.e. normal distributions
for parameter in parameters:
    name = parameter[0].name
    if name not in priors.keys():
        priors[name] = default_prior_shape

# Obtain mask of sampled parameters to run simulation in the likelihood function
parameters_idxs = [model.parameters.index(parameter[0]) for parameter in parameters]
rates_mask = [i in parameters_idxs for i in range(len(model.parameters))]
param_values = [p.value for p in model.parameters]

out_file = open("run_pydream_"+model_file, 'w')
print("Writing to PyDREAM run script: run_pydream_{}".format(model_file))
out_file.write("from pydream.core import run_dream\n")
out_file.write("from pysb.simulator import ScipyOdeSimulator\n")
out_file.write("import numpy as np\n")
out_file.write("from pydream.parameters import SampledParam\n")
#out_file.write("from pydream.convergence import Gelman_Rubin")
out_file.write("from scipy.stats import norm,uniform\n")
#out_file.write("import inspect\n")
#out_file.write("import os.path\n")
out_file.write("from "+model_module_name+" import model\n")
out_file.write("\n")
out_file.write("# DREAM Settings\n")
out_file.write("# Number of chains - should be at least 3.\n")
out_file.write("nchains = 5\n")

out_file.write("# Number of iterations\n")
out_file.write("niterations = 50000\n")
out_file.write("\n")
out_file.write("#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.\n")
out_file.write("tspan = np.linspace(0,10, num=100)\n")
out_file.write("solver = ScipyOdeSimulator(model, tspan)\n")
out_file.write("parameters_idxs = " + str(parameters_idxs)+"\n")
out_file.write("rates_mask = " + "[i in parameters_idxs for i in range(len(model.parameters))]\n" )
out_file.write("param_values = np.array([p.value for p in model.parameters])\n" )
out_file.write("\n")
out_file.write("# USER must add commands to import/load any experimental data for use in the likelihood function!\n")
out_file.write("experiments_avg = np.load()\n")
out_file.write("experiments_sd = np.load()\n")
out_file.write("like_data = norm(loc=experiments_avg, scale=experiments_sd)\n")
out_file.write("# USER must define a likelihood function!\n")
out_file.write("def likelihood(position):\n")
out_file.write("    Y=np.copy(position)\n")
out_file.write("    param_values[rates_mask] = 10 ** Y\n")
out_file.write("    sim = solver.run(param_values).all\n")
out_file.write("    logp_data = np.sum(like_data.logpdf(sim['observable']))\n")
out_file.write("    return logp_data\n")
out_file.write("\n")
#write the sampled params lines
out_file.write("sampled_params_list = list()\n")
for parameter in parameters:
    name = parameter[0].name
    value = parameter[0].value
    prior_shape = priors[name]
    print("Will sample parameter {} with {} prior around {}".format(name, prior_shape, value))
    if prior_shape == 'uniform':
        line = write_uniform_param(name, value)
        ps_name = line.split()[0]
        out_file.write(line)
        out_file.write("sampled_params_list.append({})\n".format(ps_name))
    else:
        line = write_norm_param(name, value)
        ps_name = line.split()[0]
        out_file.write(line)
        out_file.write("sampled_params_list.append({})\n".format(ps_name))

if use_GR_converge:
    out_file.write("converged = False\n")
    out_file.write("sampled_params, log_ps = run_dream(parameters=sampled_params_list,\n")
    out_file.write("                                   likelihood=likelihood,\n")
    out_file.write("                                   niterations=niterations,\n")
    out_file.write("                                   nchains=nchains,\n")
    out_file.write("                                   multitry=False,\n")
    out_file.write("                                   gamma_levels=4,\n")
    out_file.write("                                   adapt_gamma=True,\n")
    out_file.write("                                   history_thin=1,\n")
    out_file.write("                                   model_name=\'dreamzs_5chain\',\n")
    out_file.write("                                   verbose=True)\n")
    out_file.write("total_iterations = niterations\n")
    out_file.write("# Save sampling output (sampled parameter values and their corresponding logps).\n")
    out_file.write("for chain in range(len(sampled_params)):\n")
    out_file.write("    np.save(\'dreamzs_5chain_sampled_params_chain_\' + str(chain)+\'_\'+str(total_iterations), sampled_params[chain])\n")
    out_file.write("    np.save(\'dreamzs_5chain_logps_chain_\' + str(chain)+\'_\'+str(total_iterations), log_ps[chain])\n")
    #Check convergence and continue sampling if not converged

    out_file.write("GR = Gelman_Rubin(sampled_params)\n")
    out_file.write("print('At iteration: ',total_iterations,' GR = ',GR)\n")
    out_file.write("np.savetxt(\'dreamzs_5chain_GelmanRubin_iteration_\'+str(total_iterations)+\'.txt\', GR)\n")
    out_file.write("old_samples = sampled_params\n")
    out_file.write("if np.any(GR>1.2):\n")
    out_file.write("    starts = [sampled_params[chain][-1, :] for chain in range(nchains)]\n")
    out_file.write("    while not converged:\n")
    out_file.write("        total_iterations += niterations\n")
    out_file.write("        sampled_params, log_ps = run_dream(parameters=sampled_parameter_list,\n")
    out_file.write("                                           likelihood=likelihood,\n")
    out_file.write("                                           niterations=niterations,\n")
    out_file.write("                                           nchains=nchains,\n")
    out_file.write("                                           start=starts,\n")
    out_file.write("                                           multitry=True,\n")
    out_file.write("                                           gamma_levels=4,\n")
    out_file.write("                                           adapt_gamma=True,\n")
    out_file.write("                                           history_thin=1,\n")
    out_file.write("                                           model_name=\'dreamzs_5chain\',\n")
    out_file.write("                                           verbose=False,\n")
    out_file.write("                                           restart=True)\n")


            # Save sampling output (sampled parameter values and their corresponding logps).
    out_file.write("        for chain in range(len(sampled_params)):\n")
    out_file.write("            np.save('dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])\n")
    out_file.write("            np.save('dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])\n")

    out_file.write("        old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]\n")
    out_file.write("        GR = Gelman_Rubin(old_samples)\n")
    out_file.write("        print(\'At iteration: \',total_iterations,\' GR = \',GR)\n")
    out_file.write("        np.savetxt(\'dreamzs_5chain_GelmanRubin_iteration_\' + str(total_iterations)+\'.txt\', GR)\n")

    out_file.write("        if np.all(GR<1.2):\n")
    out_file.write("            converged = True\n")

else:
    out_file.write("sampled_params, log_ps = run_dream(parameters=sampled_params_list,\n")
    out_file.write("                                   likelihood=likelihood,\n")
    out_file.write("                                   niterations=niterations,\n")
    out_file.write("                                   nchains=nchains, multitry=False,\n")
    out_file.write("                                   gamma_levels=4, adapt_gamma=True,\n")
    out_file.write("                                   history_thin=1,\n")
    out_file.write("                                   model_name=\'dreamzs_5chain\',\n")
    out_file.write("                                   verbose=True)\n")
    out_file.write("total_iterations = niterations\n")
    out_file.write("# Save sampling output (sampled parameter values and their corresponding logps).\n")
    out_file.write("for chain in range(len(sampled_params)):\n")
    out_file.write("    np.save(\'dreamzs_5chain_sampled_params_chain_\' + str(chain)+\'_\'+str(total_iterations), sampled_params[chain])\n")
    out_file.write("    np.save(\'dreamzs_5chain_logps_chain_\' + str(chain)+\'_\'+str(total_iterations), log_ps[chain])\n")

if try_plot:
    out_file.write("try:\n")
    out_file.write("    #Plot output\n")
    out_file.write("    import seaborn as sns\n")
    out_file.write("    from matplotlib import pyplot as plt\n")
    out_file.write("    total_iterations = len(old_samples[0])\n")
    out_file.write("    burnin = total_iterations/2\n")
    out_file.write("    samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :],\n")
    out_file.write("                              old_samples[2][burnin:, :], old_samples[3][burnin:, :],\n")
    out_file.write("                              old_samples[4][burnin:, :]))\n")

    out_file.write("    ndims = len(sampled_parameter_names)\n")
    out_file.write("    colors = sns.color_palette(n_colors=ndims)\n")
    out_file.write("    for dim in range(ndims):\n")
    out_file.write("    fig = plt.figure()\n")
    out_file.write("    sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)\n")
    out_file.write("    fig.savefig('fig_PyDREAM_dimension_'+str(dim))\n")

    out_file.write("except ImportError:\n")
    out_file.write("    pass\n")

out_file.close()
print("pydream_it is complete!")
print("END OF LINE.")

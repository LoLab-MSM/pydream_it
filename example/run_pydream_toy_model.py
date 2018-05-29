from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm,uniform
from toy_model import model

# DREAM Settings
# Number of chains - should be at least 3.
nchains = 5
# Number of iterations
niterations = 50000

#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0,10, num=100)
solver = ScipyOdeSimulator(model, tspan)
parameters_idxs = [0, 1, 2, 3, 4]
rates_mask = [i in parameters_idxs for i in range(len(model.parameters))]
param_values = np.array([p.value for p in model.parameters])

# USER must add commands to import/load any experimental data for use in the likelihood function!
experiments_avg = np.load()
experiments_sd = np.load()
like_data = norm(loc=experiments_avg, scale=experiments_sd)
# USER must define a likelihood function!
def likelihood(position):
    Y=np.copy(position)
    param_values[rates_mask] = 10 ** Y
    sim = solver.run(param_values).all
    logp_data = np.sum(like_data.logpdf(sim['observable']))
    return logp_data

sampled_params_list = list()
sp_k_1 = SampledParam(norm, loc=np.log10(0.002), scale=2.0)
sampled_params_list.append(sp_k_1)
sp_k_2 = SampledParam(norm, loc=np.log10(0.001), scale=2.0)
sampled_params_list.append(sp_k_2)
sp_k_4 = SampledParam(norm, loc=np.log10(0.004), scale=2.0)
sampled_params_list.append(sp_k_4)
sp_k_5 = SampledParam(uniform, loc=np.log10(0.001)-1.0, scale=2.0)
sampled_params_list.append(sp_k_5)
sp_k_3 = SampledParam(norm, loc=np.log10(0.001), scale=2.0)
sampled_params_list.append(sp_k_3)
converged = False
sampled_params, log_ps = run_dream(parameters=sampled_params_list,
                                   likelihood=likelihood,
                                   niterations=niterations,
                                   nchains=nchains,
                                   multitry=False,
                                   gamma_levels=4,
                                   adapt_gamma=True,
                                   history_thin=1,
                                   model_name='dreamzs_5chain',
                                   verbose=True)
total_iterations = niterations
# Save sampling output (sampled parameter values and their corresponding logps).
for chain in range(len(sampled_params)):
    np.save('dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
    np.save('dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])
GR = Gelman_Rubin(sampled_params)
print('At iteration: ',total_iterations,' GR = ',GR)
np.savetxt('dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)
old_samples = sampled_params
if np.any(GR>1.2):
    starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
    while not converged:
        total_iterations += niterations
        sampled_params, log_ps = run_dream(parameters=sampled_parameter_list,
                                           likelihood=likelihood,
                                           niterations=niterations,
                                           nchains=nchains,
                                           start=starts,
                                           multitry=True,
                                           gamma_levels=4,
                                           adapt_gamma=True,
                                           history_thin=1,
                                           model_name='dreamzs_5chain',
                                           verbose=False,
                                           restart=True)
        for chain in range(len(sampled_params)):
            np.save('dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
            np.save('dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])
        old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
        GR = Gelman_Rubin(old_samples)
        print('At iteration: ',total_iterations,' GR = ',GR)
        np.savetxt('dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)
        if np.all(GR<1.2):
            converged = True
try:
    #Plot output
    import seaborn as sns
    from matplotlib import pyplot as plt
    total_iterations = len(old_samples[0])
    burnin = total_iterations/2
    samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :],
                              old_samples[2][burnin:, :], old_samples[3][burnin:, :],
                              old_samples[4][burnin:, :]))
    ndims = len(sampled_parameter_names)
    colors = sns.color_palette(n_colors=ndims)
    for dim in range(ndims):
    fig = plt.figure()
    sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
    fig.savefig('fig_PyDREAM_dimension_'+str(dim))
except ImportError:
    pass

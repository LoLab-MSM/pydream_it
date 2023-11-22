from pysb import *

# Empty model
Model()

# Define Monomers
# Enzyme
Monomer('E', ['binding1'])
# Substrate 1
Monomer('S1', ['binding', 'state'], {'state': ['sub', 'pro']})
# Substrate 2
Monomer('S2', ['binding', 'state'], {'state': ['sub', 'pro']})

# Initial Conditions
Parameter('E_0', 1000)
Parameter('S1_0', 500)
Parameter('S2_0', 600)
Initial(E(binding1=None), E_0)
Initial(S1(binding=None, state='sub'), S1_0)
Initial(S2(binding=None, state='sub'), S2_0)

# Observables
Observable('enzyme_total', E())
Observable('enzyme_free', E(binding1=None))
Observable('substrate_1', S1(binding=None, state='sub'))
Observable('substrate_2', S2(binding=None, state='sub'))
Observable('complex_1', E(binding1=1) % S1(binding=1, state='sub'))
Observable('complex_2', E(binding1=1) % S2(binding=1, state='sub'))
Observable('product_1', S1(binding=None, state='pro'))
Observable('product_2', S2(binding=None, state='pro'))
Observable('product_total', S1(state='pro') + S2(state='pro'))

# Define the binding rules
Parameter('k_1', 0.002)
Parameter('k_2', 0.001)
Rule('binding_1',
     E(binding1=None) + S1(state='sub', binding=None) | E(binding1=1) % S1(state='sub', binding=1), k_1, k_2)
Parameter('k_4', 0.004)
#PYDREAM_IT prior k_5 uniform 2
Parameter('k_5', 0.001)
Rule('binding_2',
     E(binding1=None) + S2(state='sub', binding=None) | E(binding1=1) % S2(state='sub', binding=1), k_4, k_5)

# Catalyze
Parameter('k_3', 0.1)
#PYDREAM_IT no-sample k_6
Parameter('k_6', 0.1)
Rule('catalyze_1', E(binding1=1) % S1(state='sub', binding=1) >> E(binding1=None) + S1(state='pro', binding=None), k_3)
Rule('catalyze_2', E(binding1=1) % S2(state='sub', binding=1) >> E(binding1=None) + S2(state='pro', binding=None), k_6)

if __name__ == "__main__":
    from pysb.simulator import ScipyOdeSimulator
    import numpy as np
    import matplotlib.pyplot as plt

    tspan = np.linspace(0, 30, num=31)  # 601
    solver = ScipyOdeSimulator(model, tspan=tspan, verbose=True)
    output = solver.run()

    color = {}
    for obs in model.observables:
        p = plt.plot(tspan, output.observables[obs.name], lw=2, label=obs.name)
        color[obs.name] = p[0].get_color()

    # Create synthetic data
    from scipy.stats import norm
    synth_obs = ['complex_2', 'product_total']
    synth_data = {}
    for obs in synth_obs:
        # 'random_state' is the random seed. Set to None if you want to create a different set of data.
        synth_data[obs] = norm.rvs(output.observables[obs], 0.1 * output.observables[obs], random_state=10)
        plt.plot(tspan, synth_data[obs], '*', ms=8, color=color[obs])
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(loc=0, bbox_to_anchor=[1, 1])
    plt.tight_layout()

    # Output data to csv files, if desired
    output_to_file = False  # Set to True if you want to save the synthetic data for calibration
    if output_to_file:
        out_file_mean = open("%s_exp_data_avg.csv" % model.name, 'w')
        out_file_sdev = open("%s_exp_data_sd.csv" % model.name, 'w')
        out_file_time = open('%s_exp_data_time.csv' % model.name, 'w')
        # write headers
        out_file_time.write("time\n")
        for j, obs in enumerate(synth_obs):
            if j > 0:
                out_file_mean.write(",")
                out_file_sdev.write(",")
            out_file_mean.write(obs)
            out_file_sdev.write(obs)
        out_file_mean.write("\n")
        out_file_sdev.write("\n")
        # write rest of output
        for i in range(len(tspan)):
            out_file_time.write("%g\n" % tspan[i])
            for j, obs in enumerate(synth_obs):
                if j > 0:
                    out_file_mean.write(",")
                    out_file_sdev.write(",")
                out_file_mean.write("%g" % synth_data[obs][i])
                # At time 0, need to make sure sdev is NON-ZERO
                val_obs_t = output.observables[obs][i] if i > 0 else min(output.observables[obs][1:])
                out_file_sdev.write("%g" % (0.1 * val_obs_t))
            out_file_mean.write("\n")
            out_file_sdev.write("\n")
        # close files
        out_file_mean.close()
        out_file_sdev.close()
        out_file_time.close()

    plt.show()

from pysb import *
from pysb.macros import *

# Empty model
Model()

#Define Monomers
# Enzyme
Monomer('E', ['binding1'])
# Substrate 1
Monomer('S1', ['binding', 'state'], {'state': ['sub', 'pro']})
# Substrate 2
Monomer('S2', ['binding', 'state'], {'state': ['sub', 'pro']})
# Product - common to both substrates
Monomer('P')
#Define the binding rules
Parameter('k_1', 0.002)
Parameter('k_2', 0.001)
Rule('binding_1', E(binding1=None) + S1(state='sub', binding=None) <> E(binding1=1) % S1(state='sub', binding=1), k_1, k_2)
Parameter('k_4', 0.004)
#PYDREAM_IT prior k_5 uniform
Parameter('k_5', 0.001)
Rule('binding_2', E(binding1=None) + S2(state='sub', binding=None) <> E(binding1=1) % S2(state='sub', binding=1), k_4, k_5)

#Catalyze
Parameter('k_3', 0.001)
#PYDREAM_IT no-sample k_6
Parameter('k_6', 0.001)
Rule('catalyze_1', E(binding1=1) % S1(state='sub', binding=1) >> E(binding1=None) + P(), k_3)
Rule('catalyze_2', E(binding1=1) % S2(state='sub', binding=1) >> E(binding1=None) + P(), k_6)

# Initial Conditions
Parameter('E_0', 1000)
Parameter('S1_0', 500)
Parameter('S2_0', 500)
Parameter('P_0', 0)
Initial(E(binding1=None), E_0)
Initial(S1(binding=None, state='sub'), S1_0)
Initial(S2(binding=None, state='sub'), S2_0)
Initial(P(), P_0)
# Observables
Observable('enzyme_total', E())
Observable('enzyme_free', E(binding1=None))
Observable('substrate_1', S1(binding=None, state='sub'))
Observable('substrate_2', S1(binding=None, state='sub'))
Observable('complex_1', E(binding1=1) % S1(binding=1, state='sub'))
Observable('complex_2', E(binding1=1) % S2(binding=1, state='sub'))
Observable('product_1', S1(binding=None, state='pro'))
Observable('product_2', S2(binding=None, state='pro'))
Observable('product_total', P())
#Expression('product_total', product_1 + product_2)

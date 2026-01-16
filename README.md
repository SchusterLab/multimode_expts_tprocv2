# multimode_expts_tprocv2
multimode  but v2



## Handling prepulses 
prepulse = {"pulse_name" : pulse_dict (containing  gain,...)} 
if pulse name is already something general like "pi_qubit_ge", don't need to specify the pulse dict, pulls directly from config. however if you want to change phase, need to change code so that requires minimal effort.

## Sweeping parameters 
self.params[param_name] = {"label": param_values.label, 
                                           "param": param_values.param, 
                                           "param_type": param_values.param_type}
label should be pulse or time tag where the qick variable is applied to 
# multimode_expts_tprocv2
multimode  but v2



## Handling Prepulses

Prepulses allow users to apply additional pulses before the main experiment pulse. These can be specified in the `cfg.expt` dictionary under the `prepulse` key. The structure for defining prepulses is as follows:

```python
prepulse = {
    "pulse_name": {
        "chan": <channel>,
        "freq": <frequency>,
        "gain": <gain>,
        "phase": <phase>,
        "length": <length>,
        "type": <pulse_type>,
        "sigma": <sigma>,  # Optional, for Gaussian pulses
        "ramp_sigma": <ramp_sigma>  # Optional, for ramped pulses
    },
    ...
}
```

### Notes:
- If the `pulse_name` corresponds to a predefined pulse (e.g., `"pi_qubit_ge"`), the system will use the default configuration from the hardware setup unless overridden.
- To modify specific parameters (e.g., `phase`), include the parameter in the `prepulse` dictionary.
- Ensure all required fields (`chan`, `freq`, `gain`, `phase`, `length`, `type`) are specified for custom pulses.

## Sweeping Parameters

Sweeping parameters in experiments can be defined using the `self.params` dictionary. The structure is:

```python
self.params[param_name] = {
    "label": <label>,
    "param": <parameter_to_sweep>,
    "param_type": <"pulse" or "time">
}
```

### Definitions:
- `label`: A descriptive name for the parameter being swept (e.g., "pulse frequency").
- `param`: The specific parameter to sweep (e.g., `"freq"`, `"gain"`, `"length"`).
- `param_type`: Indicates whether the parameter is associated with a pulse (`"pulse"`) or a timing variable (`"time"`).

### Example:
To sweep the frequency of a pulse:

```python
self.params["freq_sweep"] = {
    "label": "Pulse Frequency",
    "param": "freq",
    "param_type": "pulse"
}
```

### Label Requirements for Sweeping Parameters

When defining sweeping parameters in the `self.params` dictionary, the `label` field must match one of the following:

1. **Pulse Name Tag**:
   - If the parameter being swept is associated with a pulse (e.g., frequency, gain, length), the `label` must match the name tag of the pulse as defined in the `make_pulse` method.
   - Example:
     ```python
     self.params["freq_sweep"] = {
         "label": "main_pulse",  # Matches the pulse name tag
         "param": "freq",
         "param_type": "pulse"
     }
     ```

2. **Delay Auto Tag**:
   - If the parameter being swept is a timing variable, the `label` must match the tag used in the `delay_auto` method.
   - Example:
     ```python
     self.params["time_sweep"] = {
         "label": "wait_prepulse",  # Matches the delay_auto tag
         "param": "time",
         "param_type": "time"
     }
     ```

### Additional Details on Prepulses

The `initialize_waveforms` method in `MMProgram` handles the creation of both base pulses and prepulses. Here’s how prepulses are processed:

1. **Base Pulses**:
   - Predefined pulses like `pi_qubit_ge` are initialized with default parameters from the hardware configuration.

2. **Custom Prepulses**:
   - If `prepulse` is specified in `cfg.expt`, it should be a dictionary where each key is the name of the prepulse, and the value is another dictionary containing the pulse parameters.
   - Supported parameters for each prepulse:
     - `freq`: Frequency of the pulse.
     - `chan`: Channel on which the pulse is applied.
     - `sigma`: Standard deviation for Gaussian pulses.
     - `length`: Length of the pulse.
     - `gain`: Amplitude of the pulse.
     - `type`: Type of the pulse (`"gauss"`, `"flat_top"`, etc.).
     - `n_sigmas`: (Optional) Number of sigmas for Gaussian pulses (default: 4).
     - `ramp_sigma`: (Optional) Ramp sigma for flat-top pulses (default: 0.02).

3. **Initialization Process**:
   - Prepulses are added to the `pulses_to_load` dictionary.
   - If a prepulse name already exists in the base pulses, it is skipped to avoid overwriting.
   - The `make_pulse` method is called for each pulse, creating the waveform for later use.

### Example Configuration for Prepulses

```python
prepulse = {
    "custom_pulse": {
        "freq": 5.0e9,
        "chan": 1,
        "sigma": 0.1,
        "length": 0.4,
        "gain": 0.8,
        "type": "gauss",
        "n_sigmas": 6,
        "ramp_sigma": 0.03
    },
    "pi_qubit_ge": {}  # Uses default parameters
}
```

### Notes:
- Ensure all required fields are specified for custom pulses.
- Default parameters are used for predefined pulses unless overridden.
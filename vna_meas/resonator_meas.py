# Pseudocode for resonator power sweep
# 0. Start with general averaging parameters, locations, etc.
# 1. Wide, fast scan
# 2. Fit fr, kappa, Q
# 3. Set averages and scan width, center
# 4. Loop
#   4a. Run scan
#   4b. Fit fr, kappa, Q
#   4c. Configure next power down.

ave_exp = 1
config = {
    "npoints": 500,
    "center_freq": float(freq_center),
    "gain": gain,
    "span": float(span) * 1.3,
    "reps": int(curr_avg),
    "bw": bw,
    "delay": delay,
    "reps2": 4,
}

# Run scan here

fitparams = [min_freq, 10, 10, 0, np.max(data["amps"]), 0]
pars, err, pinit = fitter.fithanger(data["xpts"], data["amps"], fitparams=fitparams)
pars, err, pinit = fitter.fithanger(data["xpts"], data["amps"], fitparams=pars)

rmeas.plot_scan(
    data["xpts"], data["amps"], data["phases"], pars, pinit, power, slope=slope
)

freq_center = pars[0]
q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
span = pars[0] / q * inc

fitparams = [min_freq, pars[1], pars[2], pars[3], np.max(amps), 0]
pars, err, pinit1 = fitter.fithanger(xpts, amps, fitparams=fitparams)
# fitparams = [pars[0], pars[1],pars[2], pars[3], np.max(amps)]
pars, err, pinit2 = fitter.fithanger(xpts, amps, fitparams=pars)
pars_list[-1].append(pars)
rmeas.plot_scan(xpts, amps, phs, pars, pinit1, power, slope=slope)

q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4

kappa = pars[0] / q
# Power in to device, based on frequency
# Number of photons in resonator
nph = n(power**ave_exp, pars[0] * 1e6, q, pars[2], -50)
nph_list[-1].append(n(-35, pars[0] * 1e6, q, pars[2], -50))
if i == 0:
    nph0 = nph
print(nph * curr_avg * reps2 / 1.3e7)
new_avg = np.round(2e9 / nph / reps2)
print(new_avg)
spans[j] = kappa * inc

# 11.3 Classical gases with a microscopic thermometer

## Description
Files contain Python code that analyse the questions for project 11.3 that require programming. The code uses the Python libraries numpy (for mathematical calaculations), matplotlib (plotting graphs) and pandas (reading data from .dat file).

 The files used are detailed below:


### monte_carlo_acceptance.py
Runs the Monte Carlo acceptance/rejection for the 3D ideal gas case.

### non_relativistic.py
Plots histograms for multiple N_sweeps and the parameter epsilon.

### non_rel_temperature.py
Plots the histogram for E_d and also fits the theoretical probability distribution to it to see if they are consistent with each other. Estimates of T are made using both methods, quoted with their uncertainties.

### non rel single particle.py
Plots the single particle energy histogram instead this time and also the Maxwell distribution to examine whether these are consistent. 

### random momentum ideal.py
A new Monte Carlo function is now created that allows the inital momenta to be randomised rather than be fixed to e_1 as before. Plots histograms to compare both cases. More histograms for various values of the parameter a followed by a curve fit of the temperature as a function of a.

### 2d relativistic.py
A new Monte Carlo function that now calculates the energy differently for a 2D ultra-relativistic gas.

### 3d relativistic.py 
A new Monte Carlo function that now calculates the energy differently for a 3D relativistic gas.

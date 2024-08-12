from monte_carlo_acceptance import monte_carlo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 100
N_sweeps = 1000
N_updates = N * N_sweeps
energy_therm, energy_single_particle = monte_carlo(N,N_sweeps,0.1)

### Extracting data from the histogram, using them to find the temperature using curve fit, then plot both
ydata, bins = np.histogram(energy_therm, bins=100, density=True)
xdata = (bins[:-1] + bins[1:]) / 2

def theoretical_model(E, T):   # the prob distribution in eqn (1)
    return (1 / T) * np.exp(-E / T)

popt, pcov = curve_fit(theoretical_model, xdata, ydata, 1)
T = popt[0]
T_std_error = np.sqrt(pcov[0, 0])

thermom_energy_values = np.linspace(min(energy_therm), max(energy_therm), 1000) # data to plot the curve now
prob_density_values = theoretical_model(thermom_energy_values, T)

plt.hist(energy_therm, bins = 50, density = True, alpha=0.7, label='Histogram')
plt.plot(thermom_energy_values, prob_density_values, color = 'r', label = f'Theoretical model with T = {T:f}')
plt.title('Comparison of Histogram to Theoretical Model')
plt.xlabel('$E_d$', fontsize = 12)
plt.ylabel('Probability density', fontsize = 12)
plt.legend()
# plt.savefig('Temperature estimates comparison.png', dpi=300)
plt.show()

print(f'Curve fitting method T = {T}')
print(f'Error = {T_std_error}')

average = np.mean(energy_therm)
error_average = np.std(energy_therm)

print('\n')
print(f'Average method T = {average}')
print(f'Error = {error_average / np.sqrt(N_updates)}') # sigma / root(N_updates)
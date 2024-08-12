import numpy as np
import matplotlib.pyplot as plt
from monte_carlo_acceptance import monte_carlo

energy_therm, energy_single_particle = monte_carlo(100,1000,0.1)

### Plotting Maxwell-Boltzmann distribution from theory
T = np.mean(energy_therm)
print(T)

def maxwell_boltzmann(E):
    return 2 * np.pi * E**(0.5) * 1/(np.pi * T)**(1.5) * np.exp(-E / T)

E_values = np.linspace(min(energy_single_particle), max(energy_single_particle), 1000)
plt.hist(energy_single_particle, bins=50, alpha = 0.5, density=True, label='single particle energy')
plt.plot(E_values, maxwell_boltzmann(E_values), 'r', label = 'Maxwell distribution')
plt.title('Distribution of $E_i$')
plt.xlabel('$E_i$', fontsize = 12)
plt.ylabel('$f\, (E_i)$', fontsize = 12)
plt.legend()
# plt.savefig('Single_maxwell.png',dpi=300)
plt.show()
print(np.mean(energy_therm))
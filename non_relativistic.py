import numpy as np
import matplotlib.pyplot as plt
from monte_carlo_acceptance import monte_carlo

eps = 0.1
energy_therm_10, energy_single_particle_10 = monte_carlo(100,10,eps)
energy_therm_100, energy_single_particle_100 = monte_carlo(100,100,eps)
energy_therm_1000, energy_single_particle_1000 = monte_carlo(100,1000,eps)

thermom_energies = [energy_therm_10, energy_therm_100, energy_therm_1000]
number_of_sweeps = [10,100,1000]

for i in range(0,3):

    plt.hist(thermom_energies[i], bins=50, density=True)
    plt.title(f'$N_{{sweeps}} = {number_of_sweeps[i]}$')
    plt.xlabel('$E_d$', fontsize = 12)
    plt.ylabel('$f\, (E_d)$', fontsize = 12)
    # plt.savefig(f'E_d histograms {number_of_sweeps[i]} sweeps.png', dpi=300, bbox_inches='tight')
    plt.show()

a1, b1 = monte_carlo(100,1000,5)
a2, b2 = monte_carlo(100,1000,0.1)
a3, b3 = monte_carlo(100,1000,0.01)

thermom_energies = [a1, a2, a3]
epsilon_number = [5,0.1,0.01]

for i in range(0,3):

    plt.hist(thermom_energies[i], bins=50, color='green', density=True)
    plt.title(f'$\epsilon = {epsilon_number[i]}$')
    plt.xlabel('$E_d$', fontsize = 12)
    plt.ylabel('$f\, (E_d)$', fontsize = 12)
    # plt.savefig(f'Varying epsilon {epsilon_number[i]} sweeps.png', dpi=300, bbox_inches='tight')
    plt.show()
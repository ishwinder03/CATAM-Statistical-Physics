import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random_momentum_ideal import monte_carlo_random
import importlib
relativistic_module = importlib.import_module("2d_relativistic")
monte_carlo_random_2d = relativistic_module.monte_carlo_random_2d

def monte_carlo_random_3d(N, N_sweeps, epsilon,a):
    N_updates = N * N_sweeps

    ### STEP 1: intialise momentum for all N particles as e_1 and initialise E_d
    p_i = []

    for i in range(0,N):
        p = np.random.uniform(-a,a,3)  # set random momentum each particle
        p_i.append(p)

    p_i = np.array(p_i) 
    E_d = 0

    single_particle_energy_values = []
    E_d_values = []
    
    ### STEP 2: randomising N_updates amount of times
    for n in range(int(N * N_sweeps)):
        index = np.random.randint(N)
        p_curr = p_i[index]
        E_curr = np.sqrt(1 + np.dot(p_curr, p_curr)) - 1   # computing current energy of particle E_curr

        delta_p = np.random.uniform(-epsilon, epsilon, 3)  # random perturbation
        p_prop = p_curr + delta_p
        E_prop = np.sqrt(1 + np.dot(p_prop, p_prop)) - 1    # new proposed energy after perturbation

        delta_E = E_prop - E_curr

        ### STEP 3: accept or reject
        if delta_E <= E_d:  # accept
            p_i[index] = p_prop
            E_d -= delta_E
        
        # reject so do nothing, if statement is ignored
        E_d_values.append(E_d)
        single_particle_energy_values.append(E_curr)

    return np.array(E_d_values), np.array(single_particle_energy_values), p_i


energy_therm, energy_single_particle, mom = monte_carlo_random_3d(100, 1000, 0.1, 1) # a = 1

print(f'Temperature of gas = {np.mean(energy_therm)}')

fig, ax = plt.subplots(figsize = (11,5), nrows = 1, ncols = 2)

ax[0].hist(energy_therm, bins = 50, density = True)
ax[0].set_title('$E_d$ distribution', fontsize = 12)
ax[0].set_ylabel('$f \, (E_d)$', fontsize = 12)
ax[0].set_xlabel('$E_d$', fontsize = 12)

ax[1].hist(energy_single_particle, bins = 50, density = True, color = 'green')
ax[1].set_title('$E_i$ distribution', fontsize = 12)
ax[1].set_ylabel('$f \, (E_i)$', fontsize = 12)
ax[1].set_xlabel('$E_i$', fontsize = 12)

# plt.savefig('rel a=1', dpi = 300, bbox_inches='tight')
plt.show()

a_values = np.linspace(0.1,2,100)
temperatures = np.zeros_like(a_values)
total_energies = np.zeros_like(temperatures)

for i in range(0, len(a_values)):
    energy_therm2, energy_single_particle2, mom2 = monte_carlo_random_3d(100, 1000, a_values[i]/10, a_values[i])
    current_temp = np.mean(energy_therm2)
    temperatures[i] = current_temp

    ### Calc total energy by from E_tot = (N * E_i) + E_d, determine E_i from the final momentum of each particle
    particle_energies = np.zeros(100)
    for j in range(0, 100):
        current_energy = np.sqrt(abs(np.dot(mom2[j], mom2[j])))
        particle_energies[j] = current_energy

    total_energy = np.sum(particle_energies)
    total_energies[i] = total_energy


def theoretical_model(x,alpha,beta):   # the prob distribution in eqn (1)
    return alpha * x**beta

popt, pcov = curve_fit(theoretical_model, total_energies, temperatures)
alpha_fit = popt[0]
beta_fit = popt[1]
alpha_std_error = np.sqrt(pcov[0, 0])
curve_fit_data = theoretical_model(total_energies, alpha_fit, beta_fit)

plt.scatter(total_energies, temperatures, alpha = 0.5, marker = 'x', label = 'Monte Carlo data')
plt.plot(total_energies, curve_fit_data, color = 'r', label = f'Curve fit, T = ${alpha_fit:.6f} \, E_{{tot}}^{{{beta_fit:.2f}}}$')
plt.title('Temperature as a function of total energy')
plt.xlabel('$E_{{tot}}$', fontsize = 12)
plt.ylabel('$T$', fontsize = 12)
plt.legend()
# plt.savefig('3D Temperature dependence on E_tot', dpi=300, bbox_inches='tight')
plt.show()

print(alpha_fit)
print(beta_fit)
print(alpha_std_error)

energy_therm_ideal, energy_single_particle_ideal = monte_carlo_random(100, 1000, 0.1, 1)
energy_therm_2d, energy_single_particle_2d, mom_2d = monte_carlo_random_2d(100, 1000, 0.1, 1)
energy_therm_3d, energy_single_particle_3d, mom_3d = monte_carlo_random_3d(100, 1000, 0.1, 1)

A = 0.9
b = 60

fig, ax = plt.subplots(figsize=(6,12), nrows = 3)
ax[0].hist(energy_single_particle_3d, bins=b, alpha = A, density=True, label='3D relativistic gas')
ax[1].hist(energy_single_particle_2d, bins=b, alpha = A, density=True, label='2D ultra-relativistic gas', color='C1')
ax[2].hist(energy_single_particle_ideal, bins=b, alpha = A, density=True, label='3D ideal gas', color='C2')

ax[0].set_title('Distribution of $E_i$ for different gases')
ax[2].set_xlabel('$E_i$', fontsize = 12)
ax[0].set_xlim(right = 3)
ax[1].set_xlim(right = 3)
ax[2].set_xlim(right = 3)
ax[0].set_ylabel('$f\, (E_i)$', fontsize = 12)
ax[1].set_ylabel('$f\, (E_i)$', fontsize = 12)
ax[2].set_ylabel('$f\, (E_i)$', fontsize = 12)
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.tight_layout()
plt.savefig('final comparison',dpi=300, bbox_inches='tight')
plt.show()

print(np.mean(energy_therm_ideal), np.mean(energy_therm_2d), np.mean(energy_therm_3d))
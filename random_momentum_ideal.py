import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from monte_carlo_acceptance import monte_carlo

def monte_carlo_random(N, N_sweeps, epsilon,a):
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
        E_curr = np.dot(p_curr, p_curr) / 2 # computing current energy of particle E_curr

        delta_p = np.random.uniform(-epsilon, epsilon, 3)  # random perturbation
        p_prop = p_curr + delta_p
        E_prop = np.dot(p_prop, p_prop) / 2  # new proposed energy after perturbation

        delta_E = E_prop - E_curr

        ### STEP 3: accept or reject
        if delta_E <= E_d:  # accept
            p_i[index] = p_prop
            E_d -= delta_E
        
        # reject so do nothing, if statement is ignored
        E_d_values.append(E_d)
        single_particle_energy_values.append(E_curr)

    return np.array(E_d_values), np.array(single_particle_energy_values)

if __name__ == "__main__":


    energy_therm, energy_single_particle = monte_carlo_random(100, 1000, 0.1, 1)
    x,y = monte_carlo(100, 1000, 0.1)

    print(np.mean(energy_single_particle))

    A = 0.5
    plt.hist(energy_therm, bins=50, density=True, alpha = A, label = 'random initial momentum with $a = 1$')
    plt.hist(x, bins=50, density=True, alpha = A, label = 'fixed intial momentum')
    plt.title('$E_d$ distributions')
    plt.xlabel('$E_d$', fontsize = 12)
    plt.ylabel('$f\, (E_d)$', fontsize = 12)
    plt.legend()
    # plt.savefig('E_d random vs fixed momentum for a=1', dpi=300)
    plt.show()

    plt.hist(energy_single_particle, bins=50, density=True, alpha = A, label = 'random initial momentum with $a = 1$')
    plt.hist(y, bins=50, density=True, alpha = A, label = 'fixed intial momentum')
    plt.title('$E_i$ distributions')
    plt.xlabel('$E_i$', fontsize = 12)
    plt.ylabel('$f\, (E_i)$', fontsize = 12)
    plt.legend()
    # plt.savefig('E_i random vs fixed momentum for a=1', dpi=300)
    plt.show()

    energy_therm10, energy_single_particle10 = monte_carlo_random(100, 1000, 0.7, 7)
    energy_therm9, energy_single_particle9 = monte_carlo_random(100, 1000, 0.5, 5)
    energy_therm8, energy_single_particle8 = monte_carlo_random(100, 1000, 1, 10)
    energy_therm7, energy_single_particle7 = monte_carlo_random(100, 1000, 0.05, 0.5)
    energy_therm6, energy_single_particle6 = monte_carlo_random(100, 1000, 0.1, 1)
    energy_therm5, energy_single_particle5 = monte_carlo_random(100, 1000, 0.15, 1.5)

    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    A = 0.5
    b=60

    ax[0,0].hist(energy_therm9, bins=b, density=True, alpha=A, label='a = 5')
    ax[0,0].hist(energy_therm10, bins=b, density=True, alpha=A, label='a = 7')
    ax[0,0].hist(energy_therm8, bins=b, density=True, alpha=A, label='a = 10')
    ax[0,0].set_title('$E_d$ distributions', fontsize = 12)
    ax[0,0].set_xlabel('$E_d$', fontsize = 12)
    ax[0,0].set_xlim(-3, 100)
    ax[0,0].set_ylabel('$f \, (E_d)$', fontsize = 12)
    ax[0,0].legend(fontsize = 11)

    ax[0,1].hist(energy_single_particle9, bins=b, density=True, alpha=A, label='a = 5')
    ax[0,1].hist(energy_single_particle10, bins=b, density=True, alpha=A, label='a = 7')
    ax[0,1].hist(energy_single_particle8, bins=b, density=True, alpha=A, label='a = 10')
    ax[0,1].set_title('$E_i$ distributions', fontsize = 12)
    ax[0,1].set_xlabel('$E_i$', fontsize = 12)
    ax[0,1].set_xlim(-5, 150)
    ax[0,1].set_ylabel('$f \, (E_i)$', fontsize = 12)
    ax[0,1].legend(fontsize = 11)

    ax[1,0].hist(energy_therm7, bins=b, density=True, alpha=A, label='a = 0.5')
    ax[1,0].hist(energy_therm6, bins=b, density=True, alpha=A, label='a = 1')
    ax[1,0].hist(energy_therm5, bins=b, density=True, alpha=A, label='a = 1.5')
    ax[1,0].set_title('$E_d$ distributions', fontsize = 12)
    ax[1,0].set_xlabel('$E_d$', fontsize = 12)
    ax[1,0].set_xlim(-0.1, 2.2)
    ax[1,0].set_ylabel('$f \, (E_d)$', fontsize = 12)
    ax[1,0].legend(fontsize = 11)

    ax[1,1].hist(energy_single_particle7, bins=b, density=True, alpha=A, label='a = 0.5')
    ax[1,1].hist(energy_single_particle6, bins=b, density=True, alpha=A, label='a = 1')
    ax[1,1].hist(energy_single_particle5, bins=b, density=True, alpha=A, label='a = 1.5')
    ax[1,1].set_title('$E_i$ distributions', fontsize = 12)
    ax[1,1].set_xlabel('$E_i$', fontsize = 12)
    ax[1,1].set_xlim(-0.1, 2.5)
    ax[1,1].set_ylabel('$f \, (E_i)$', fontsize = 12)
    ax[1,1].legend(fontsize = 11)

    plt.tight_layout()
    plt.savefig('E_i_random_vs_fixed_momentum_for_different_a.png', dpi=300, bbox_inches='tight')
    plt.show()

    N = 100
    N_sweeps = 1000
    a_values = np.linspace(0,10,100)
    temperatures = np.zeros_like(a_values)

    for i in range(0, len(a_values)):
        energy_therm2, energy_single_particle2 = monte_carlo_random(100, 1000, a_values[i]/10, a_values[i])
        current_temp = np.mean(energy_therm2)
        temperatures[i] = current_temp

    def theoretical_model(x,C):   # the prob distribution in eqn (1)
        return C * x**2

    popt, pcov = curve_fit(theoretical_model, a_values, temperatures)
    C_fit = popt[0]
    C_std_error = np.sqrt(pcov[0, 0])

    plt.scatter(a_values, temperatures, alpha=0.5, marker = 'x', label = 'Monte Carlo data')
    plt.plot(a_values, theoretical_model(a_values, C_fit),'r', label = f'Curve fit, T = ${C_fit:4f} \, a^2$')
    plt.title('Temperature dependence on a')
    plt.xlabel('$a$', fontsize = 12)
    plt.ylabel('$T$', fontsize = 12)
    plt.legend()
    # plt.savefig('Temperature dependence on a', dpi=300)
    plt.show()

    print(C_fit)
    print(C_std_error)

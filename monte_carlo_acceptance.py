import numpy as np

def monte_carlo(N, N_sweeps, epsilon):
    N_updates = N * N_sweeps

    ### STEP 1: intialise momentum for all N particles as e_1 and initialise E_d
    p = np.array([1.,0.,0.])  # set momentum to e_1 for one particle
    p_i = []

    for i in range(0,N):
        p_i.append(p)

    p_i = np.array(p_i) # momentum set to e_1 for all particles
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
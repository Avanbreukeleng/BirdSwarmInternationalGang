import numpy as np
from main import Bird
import dill
import time
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=sys.maxsize) fixes an error on Alejandro's computer, should not be necessary

dill.settings['recurse'] = True  # Enables recursive pickling
dill.settings['protocol'] = -1   # Use the highest available protocol


# list of the parameters in a Px8 matrix where P is the number of sets of parameters
# to select one set of parameters select inparam[:,X] where x is one of the columns (=set of param)
# A set of param is of the form np.array (seed, vel, N, R, L, eta, dt, Nstep)

class ParameterModifier:
    def __init__(self, inparam):
        self.parameters = np.array([inparam])

    def modify_parameters(self, new_N, new_eta):
        for i in range(len(new_N)):  # Creates a new parameter set with modified N and add it to the array of all parameter sets
            modified_parameters = np.copy(self.parameters[-1])
            modified_parameters[2] = new_N[i]
            self.parameters = np.vstack((self.parameters, modified_parameters))

        for j in range(len(new_eta)): # Creates a new parameter set with modified eta and add it to the array of all parameter sets
            modified_parameters = np.copy(self.parameters[-1])
            modified_parameters[5] = new_eta[j]
            self.parameters = np.vstack((self.parameters, modified_parameters))

    def get_parameters(self):
        return self.parameters

#For scaling: density rho = 4
# How to use ParameterModifier:
inparam = np.array([1, 0.033, 2, 1, 20, 1, 1, 500])  # first define a set of parameters, # (seed, vel, N, R, L, eta, dt, Nstep)
# new sets will be based on this inital one, with different values of eta or N
parameter_modifier = ParameterModifier(inparam)  # Calls the param modifier class
new_N = np.array([]) # If the number of birds should stay constant, just use an empty array
# new_N = np.linspace(10, 4000, num=0) # Here N is the number of birds
new_eta = np.linspace(0,6.3,63) # Eta is the noise
parameter_modifier.modify_parameters(new_N, new_eta)
resulting_params = parameter_modifier.get_parameters() #Calls the final array of all inital parameters




#Now the second class: Bird_Simulator
init_phase = np.array([[1, 2, 3]])  # Initialize the phase transition parameter array to be able to stack other arrays on it.

class Bird_Simulator():  # This class' goal is to yield an array of [v_a, rho, eta] values to use for the phase transition analysis
    def __init__(self, init_phase):
        self.phase_transition_parameters = init_phase

    # NOTATION [a,b] a selects the row (individual parameter set), b selects the column (# of the paramter set)
    def run_all_bird(self, resulting_params):
        va_matrix = []  # Initialize an empty list to store va vectors, this has essentially the same purpose as init_phase, but for a different array
        self.t = np.zeros_like(resulting_params[:, 1])
        for j in range(len(resulting_params[:, 1])):  # This loops over all distinct sets of parameters
            start_time = time.time()
            print(j) # useful check to see what parameter set is currently being run, and that there are no infinite loops
            Pset = resulting_params[j, :]
            print("Pset",Pset)
            Sim1 = Bird(int(Pset[0]), Pset[1], int(Pset[2]), Pset[3], Pset[4], Pset[5], Pset[6],
                        int(Pset[7]))  # The number of birds and timestep must be an integer
            Nset = resulting_params[j, 2]
            Lset = resulting_params[j, 4]
            rho = Nset / (Lset) ** 2
            eta = resulting_params[j, 5]
            va, mean_va = Sim1.va, Sim1.mean_va # va is an array, mean_va = number
            va_matrix.append(va)  # Append the va vector to the list
            local_trans = np.array([[mean_va, rho, eta]])
            self.phase_transition_parameters = np.vstack((self.phase_transition_parameters, local_trans))
            self.t[j] = (time.time() - start_time)
        print(repr(self.t))
        self.phase_transition_parameters = np.delete(self.phase_transition_parameters, 0,
                                                     axis=0)  # Delete the first row of the phase_transition_parameters array, which was created in init_phase
        self.matrix_split = np.vsplit(self.phase_transition_parameters,
                                      [len(new_N) + 1])  # This splits the transition parameter array into two arrays,
                                                                          # one with constant eta and varying N, and one with constant N and varying eta
        self.N_matrix = self.matrix_split[0]
        self.eta_matrix = self.matrix_split[1]
        self.va_matrix = np.array(va_matrix)  # Convert the list to a numpy array
        self.va_matrix = np.delete(self.va_matrix, 0,
                                   axis=0)  # Delete the first row, which was created with an empty list


    def get_N_matrix(self):
        return self.N_matrix

    def get_eta_matrix(self):
        return self.eta_matrix

    def get_va_matrix(self):
        return self.va_matrix

# How to use Run_all birds
# Has to be used after the Parameter Modifier Class, so that resulting_params is defined
init_phase = np.array([[1, 2, 3]])  # Initialize the phase trans parameter array to be able to stack other arrays on it.
bird_sim = Bird_Simulator(init_phase)
bird_sim.run_all_bird(resulting_params)
#The following three arrays are the ones we use for plotting the order parameters
N_matrix = bird_sim.N_matrix
eta_matrix = bird_sim.eta_matrix
va_matrix = bird_sim.va_matrix

SAVE = False
if SAVE:
    # np.savetxt("Vector1.csv", swarm.vector, delimiter=",")
    name = input('Desired file name identification?\t')
    with open('New_output_files/' + name + '_fixeta_swarm.csv', 'wb') as pickle_out:
        dill.dump(bird_sim, pickle_out)
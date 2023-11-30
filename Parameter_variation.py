import numpy as np
from main import Bird
import dill

dill.settings['recurse'] = True  # Enable recursive pickling
dill.settings['protocol'] = -1   # Use the highest available protocol


# list of the parameters in a Px8 matrix where P is the number of sets of parameters
# to select one set of parameters select inparam[:,X] where x is one of the columns (=set of param)
# A set of param is of the form np.array (seed, vel, N, R, L, eta, dt, Nstep)

class ParameterModifier:
    def __init__(self, inparam):
        self.parameters = np.array([inparam])

    def modify_parameters(self, new_N, new_eta):
        for i in range(len(new_N)):
            modified_parameters = np.copy(self.parameters[-1])  # Copy the last row
            modified_parameters[2] = new_N[i]
            self.parameters = np.vstack((self.parameters, modified_parameters))

        for j in range(len(new_eta)):
            modified_parameters = np.copy(self.parameters[-1])  # Copy the last row
            modified_parameters[5] = new_eta[j]
            self.parameters = np.vstack((self.parameters, modified_parameters))

    def get_parameters(self):
        return self.parameters


# How to use ParameterModifier:
inparam = np.array([1, 0.033, 100, 1, 20, 0.1, 1, 1000])  # first set of parameters
parameter_modifier = ParameterModifier(inparam)  # Call the param modifier class
# new_N = np.linspace(40, 150, num=15) #Here N is the PLnumber of birds
new_N = np.array([])
# new_eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0])
# new_eta = np.linspace(0,5,50)
new_eta = np.linspace(0,5,50)
parameter_modifier.modify_parameters(new_N, new_eta)

resulting_params = parameter_modifier.get_parameters()
# print(resulting_params)
# End of using ParameterModifier
# Run simulation

init_phase = np.array([[1, 2, 3]])  # Initialize the phase trans parameter array to be able to stack other arrays on it.


class Bird_Simulator():  # This class' goal is to yield an array [v_a, rho, eta] to use for the phase transition analysis
    def __init__(self, init_phase):
        self.phase_transition_parameters = init_phase

    # NOTATION [a,b] a selects the line, b the column
    def run_all_bird(self, resulting_params):
        va_matrix = []  # Initialize an empty list to store va vectors, this has essentially the same purpose as init_phase, but for a different array
        for j in range(len(resulting_params[:, 1])):  # this loops over all distinct sets of parameters
            print(j)
            Pset = resulting_params[j, :]
            # print("Pset",Pset)
            Sim1 = Bird(int(Pset[0]), Pset[1], int(Pset[2]), Pset[3], Pset[4], Pset[5], Pset[6],
                        int(Pset[7]))  # This is not elegant but it works
            Nset = resulting_params[j, 2]
            Lset = resulting_params[j, 4]
            rho = Nset / (Lset) ** 2
            eta = resulting_params[j, 5]
            va, mean_va = Sim1.order_parameter_calculation()  # va is array, meanva = number
            va_matrix.append(va)  # Append the va vector to the list
            local_trans = np.array([[mean_va, rho, eta]])
            self.phase_transition_parameters = np.vstack((self.phase_transition_parameters, local_trans))
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


SAVE = True


# How to use Run_all birds
# Has to be used after the Parameter Modifier Class, so that resulting_params is defined
init_phase = np.array([[1, 2, 3]])  # Initialize the phase trans parameter array to be able to stack other arrays on it.
bird_sim = Bird_Simulator(init_phase)
bird_sim.run_all_bird(resulting_params)
#The following three arrays are the ones we use for plotting the order parameters
N_matrix = bird_sim.N_matrix
eta_matrix = bird_sim.eta_matrix
va_matrix = bird_sim.va_matrix

if SAVE:
    # np.savetxt("Vector1.csv", swarm.vector, delimiter=",")
    name = input('Desired file name identification?\t')
    with open('Output_files/' + name + '_pickle_param_swarm.csv', 'wb') as pickle_out:
        dill.dump(bird_sim, pickle_out)
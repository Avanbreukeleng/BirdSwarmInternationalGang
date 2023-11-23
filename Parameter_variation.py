
from main import Bird
import numpy as np
import matplotlib.pyplot as plt


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
            modified_parameters[4] = new_eta[j]
            self.parameters = np.vstack((self.parameters, modified_parameters))

    def get_parameters(self):
        return self.parameters


# How to use ParameterModifier:
inparam = np.array([1, 0.033, 500, 1, 10, np.pi / 4, 1, 10])  # first set of parameters
parameter_modifier = ParameterModifier(inparam)  # Call the param modifier class
new_N = np.linspace(10, 100, num=10)
new_eta = np.array([0.1, 0.5, 1.0, 2.0])
parameter_modifier.modify_parameters(new_N, new_eta)
resulting_params = parameter_modifier.get_parameters()
# print(resulting_params)
# End of using ParameterModifier

init_phase = np.array([[1, 2, 3]])  # Initialize the phase trans parameter array to be able to stack other arrays on it.


class Bird_Simulator():  # This class' goal is to yield an array [v_a, rho, eta] to use for the phase transition analysis
    def __init__(self, init_phase):
        self.phase_transition_parameters = init_phase

    # NOTATION [a,b] a selects the line, b the column
    def run_all_bird(self, resulting_params):
        for j in range(len(resulting_params[:, 1])):  # this loops over all distinct sets of parameters
            Pset = resulting_params[j, :]
            Sim1 = Bird(int(Pset[0]), Pset[1], int(Pset[2]), Pset[3], Pset[4], Pset[5], Pset[6],
                        int(Pset[7]))  # This is not elegant ;)
            Nset = resulting_params[j, 2]
            Lset = resulting_params[j, 4]
            rho = Nset / (Lset) ** 2
            eta = resulting_params[j, 5]
            va, mean_va = Sim1.order_parameter_calculation() #va is array, meanva = number
            print(va)
            local_trans = np.array([[mean_va, rho, eta]])
            self.phase_transition_parameters = np.vstack((self.phase_transition_parameters, local_trans))
        self.phase_transition_parameters = np.delete(self.phase_transition_parameters, 0,
                                                     axis=0)  # Delete the first row of the phase_transition_parameters array, which was created in init_phase

    def get_trans_parameters(self):
        return self.phase_transition_parameters

#How to use Run_all birds
#Has to be used after the Parameter Modifier Class, so that resulting_params is defined
init_phase = np.array([[1, 2, 3]])  # Initialize the phase trans parameter array to be able to stack other arrays on it.
bird_sim = Bird_Simulator(init_phase)
bird_sim.run_all_bird(resulting_params)
phase_matrix = bird_sim.get_trans_parameters()
#print(phase_matrix)



# NOTICE: phase_transition_parameters is an array [v_a,rho, eta] example [[1,2,3],[4,5,6], ...] also I AM NOT SURE IF I need to call it like self.sth
def plot_phase_transition(phase_matrix):
    fig, ax = plt.subplots()
    # Plot order parameter as a function of time
    plt.title("Evolution of Order Parameter in time")
    plt.xlabel("Time Step")
    plt.ylabel("Order Parameter")
    plt.plot(np.arange(0,len(va),1),va, '-', color = 'g')
    # Plot mean order parameter as a function of density when noise is fixed
    plt.title("Order Parameter vs Density")
    plt.xlabel("Density")
    plt.ylabel("Order Parameter")
    plt.plot(N_matrix[:,1],N_matrix[:,0],'o', color = 'r')
    # Plot mean order parameter as a function of noise when density is fixed
    plt.title('Order Parameter vs eta')
    plt.xlabel('Eta')
    plt.ylabel('Order Parameter')
    plt.plot(eta_matrix[:,2],eta_matrix[:,0],'o',color='b')
    plt.show()

PLOT = True



import numpy as np
from main import Bird
import matplotlib.pyplot as plt

#list of the parameters in a Px8 matrix where P is the number of sets of parameters
#to select one set of parameters select InParam[:,X] where x is one of the columns (=set of param)
#A set of param is of the form np.array (seed, vel, N, R, L, eta, dt, Nstep)

class ParameterModifier:
    def __init__(self, InParam):
        self.parameters = InParam

    def modify_parameters(self, new_N, new_eta): # This will modify the parameters
        for i in range(len(new_N)): #This for loop modifies N
            modified_parameters = np.array([np.copy(self.parameters[0]])
            modified_parameters[2] = new_N[i]
            self.parameters = np.vstack((self.parameters, modified_parameters)) # Vertically stack the new sets of parameters

         for j in range(len(new_eta)):  # This for loop modifies eta
            modified_parameters = np.array([np.copy(self.parameters[0]])
            modified_parameters[4] = new_eta[j]
            self.parameters = np.vstack((self.parameters, modified_parameters))

    def get_parameters(self):
        return self.parameters

#How to use ParameterModifier:
InParam = np.array(1, 0.033, 1, 1, 10, np.pi/4, 1, 10) #first set of parameters
parameter_modifier = ParameterModifier(InParam) #Call the param modifier untion
new_N = np.linspace(10, 100, num=10)
new_eta = np.array([0.1, 0.5, 1.0, 2.0])
ParameterModifier.modify_parameters(new_N, new_eta)
resulting_params = ParameterModifier.get_parameters()
print(resulting_params)
#End of using ParameterModifier

init_phase = np.array([[1, 2, 3]]) #Initialize the phase trans parameter array to be able to stack other arrays on it.
#TODO make this an object
def Run_all_Bird()
    for j in (range(len(InParam)[1, :])) #this loops over all distinct sets of parameters
        Pset = Inparam[:, j]
        Sim1 = Bird(Pset)
        Nset = Inparam[2, j]
        Lset = Inparam[4, j]
        rho = Nset*(1/(Lset**2))
        eta = Inparam[5, j]
        local_trans = np.array([[v_a, rho, eta]]) #TODO CHECK THAT THIS INDEED CALLS VA FROM MAIN
        phase_transition_parameters = np.vstack((init_phase, local_trans))

phase_transition_parameters = np.delete(phase_transition_parameters, 0, axis=0) #Delete the first row of the phase_transition_parametrs array, which was created in init_phase


# NOTICE: phase_transition_parameters is an array [v_a,rho, eta] example [[1,2,3],[4,5,6], ...] also I AM NOT SURE IF I need to call it like self.sth
def plot_phase_transition(phase_transition_parameters):
# Investigate the behaviour of order parameter v_a as a function of density and eta
    fig, ax = plt.subplots()
    plt.title("Order Parameter vs Density")
    plt.xlabel("Density")
    plt.ylabel("Order Parameter")
    plt.plot(phase_transition_parameters[:,1],phase_transition_parameters[:,0],'o', color = 'r')
    plt.title('Order Parameter vs eta')
    plt.xlabel('Eta')
    plt.ylabel('Order Parameter')
    plt.plot(phase_transition_parameters[:,2],phase_transition_parameters[:,0],'o',color='b')
    plt.show()

PLOT = True

# if PLOT



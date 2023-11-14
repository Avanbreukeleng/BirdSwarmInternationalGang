import numpy as np
from main import Bird
import matplotlib.pyplot as plt

#list of the parameters in a Px8 matrix where P is the number of sets of parameters
#to select one set of parameters select InParam[:,X] where x is one of the columns (=set of param)
#A set of param is of the form np.array (seed, vel, N, R, L, eta, dt, Nstep)


InParam = np.array(1, 0.033, 1, 1, 10, np.pi/4, 1, 10) #first set of parameters


# TODO compute density and eta from this matrix
#

# NOTICE: phase_transition_parameters is an array [v_a,rho, eta] example [[1,2,3],[4,5,6], ...] also I AM NOT SURE IF I need to call it like self.sth
def plot_phase_transition(self.phase_transition_parameters):
# Investigate the behaviour of order parameter v_a as a function of density and eta
    fig, ax = plt.subplots()
    plt.title("Order Parameter vs Density")
    plt.xlabel("Density")
    plt.ylabel("Order Parameter")
    plt.plot(self.phase_transition_parameters[:,1],self.phase_transition_parameters[:,0],'-', color = 'r')
    plt.title('Order Parameter vs eta')
    plt.xlabel('Eta')
    plt.ylabel('Order Parameter')
    plt.plot (self.phase_transition_parameters[:,2],self.phase_transition_parameters[:,0],'-',color='b')
    plt.show()



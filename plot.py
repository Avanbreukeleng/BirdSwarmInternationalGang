import matplotlib.pyplot as plt
import dill

name = input('Desired file name identification?\t')
with open('Output_files/' + name + '_pickle_param_swarm.csv',"rb") as pickle_in:  # "rb" because we want to read in binary mode
    bird_sim = dill.load(pickle_in)
#Commit test














# NOTICE: phase_transition_parameters is an array [v_a,rho, eta] example [[1,2,3],[4,5,6], ...]
# def plot_phase_transition(va_matrix, N_matrix, eta_matrix):
#     # fig, ax = plt.subplots()
#     # Plot order parameter as a function of time for each set of initial parameters
#     # for i in range(len(va_matrix[:,0])):
#     #     plt.title("Evolution of Order Parameter in time for initial set %d" %i)
#     #     plt.xlabel("Time Step")
#     #     plt.ylabel("Order Parameter")
#     #     plt.plot(np.arange(0, len(va_matrix[0]), 1), va_matrix[i], '-', color='g')
#     # Plot mean order parameter as a function of density when noise is fixed
#     fig, ax = plt.subplots()
#     plt.title("Order Parameter vs Density")
#     plt.xlabel("Density")
#     plt.ylabel("Order Parameter")
#     plt.plot(N_matrix[:, 1], N_matrix[:, 0], 'o', color='r')
#     plt.show()
#     # Plot mean order parameter as a function of noise when density is fixed
#     fig, ax = plt.subplots()
#     plt.title('Order Parameter vs eta')
#     plt.xlabel('Eta')
#     plt.ylabel('Order Parameter')
#     plt.plot(eta_matrix[:, 2], eta_matrix[:, 0], 'o', color='b')
#     plt.show()
#
# plot = plot_phase_transition(bird_sim.va_matrix, bird_sim.N_matrix, bird_sim.eta_matrix)
# #PLOT = True
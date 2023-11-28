# -*- coding: utf-8 -*-

# import csv
import numpy as np

import matplotlib; matplotlib.use("TkAgg")
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import time


# from Parameter_variation import *

# import pickle
# from objects.assembly import Assembly
# from objects.mesh import Mesh
# from objects.element import Element

# from structure import geometry

    # =============================================================================
    #  PHY571 Numerical Physics Project
    # =============================================================================
    # ============================Start of preamble================================
    # This Python-function belongs to the PHY571 Numerical Physics Project. The file contains a function that
    # calculates dynamics of a swarm of interacting boids specified by the user
    #
    # The function uses a TOML file as input and returns a dictionary as output.
    #
    # Written by Pierre-Louis Wasmer, Fatemeh Nouri and Alejandro van Breukelen García,
    # M1 Physics
    # Institut Polytechnique de Paris
    # November 2023 & December 2023
    # ==============================End of Preamble================================


class Bird():

    def init_vector(self, seed, N,L):
        np.random.seed(seed)
        posvector = L*np.random.rand(N, 2)
        theta = np.random.uniform(0, 2*np.pi,(N,1))
        # print(theta)
        vector = np.hstack((posvector, theta))
        vector = np.hstack((vector, np.zeros_like(theta)))
        vector = np.hstack((vector, np.zeros_like(theta)))
        vector = vector[np.newaxis,:, :]
        return vector

    """NOTATION: VECTOR[STEP][NBIRD,XYZBxBy]"""
    # the columns go as follows: X, Y , THETA (angle of boid with respect to horizontal), BIN (bin coordinate of boids position (bx,by))

    """NOTATION: VECTOR[STEP][NBIRD,XYZ]"""
    
    def __init__(self,seed,vel,N,R,L,eta,dt,Nsteps):
        self.vector = self.init_vector(int(seed),int(N),L)   # This is the NStepsx3 array that stores x,y and theta value at each step
                                                 #At each step, add another layer to the array (in 3D)
        self.velocity = vel                      # Constant norm of velocity for all birds
        self.R = R                               # Radius of influence of each boid
        self.L = L                               # Size of the world
        self.eta = eta                           # Interval of noise in theta
        self.dt = dt                             # Constant time step
        self.N = N
        self.Nsteps = Nsteps
        self.Lbins = L/int(L/R)

        #self.rho = self.N/(self.L)**2
        self.Nbins = int(self.L/self.Lbins)
        self.update()

    def bin_update(self): #This function does not change the bins themselves, but tracks which birds are in which bin
        dummy = self.vector[-1][:,0:2]/self.Lbins
        dummy.astype(int) # round down at which bin it is, (from bin 0 to Nbin-1)
        self.vector[-1][:, 3:5] = dummy

    def evolve(self):
        # Function evolves the movement of the boids for one timestep dt and velocity v
        vector_old = self.vector[-1]
        vector_add = np.zeros_like(vector_old)
        theta = vector_old[:, 2]
        vector_add[:,0] = self.velocity*self.dt*np.cos(theta)
        vector_add[:,1] = self.velocity*self.dt*np.sin(theta)
        vector_new = vector_old + vector_add

        vector_new[:,0:2] = vector_new[:,0:2] % self.L # Periodic boundary conditions

        vector_new_reshaped = vector_new.reshape(1, self.N, 5) #reshape the new post matrix in order to be able to concatenate
        self.vector = np.concatenate((self.vector, vector_new_reshaped), axis=0)
        self.bin_update()

    def new_theta(self): #TODO make a stepby step example with N=5 maybe to explain logic
        # Function to calculate the average angle of neighbouring boids and update
        Neighbours = np.full((self.N, self.N), np.nan)
        np.fill_diagonal(Neighbours, self.vector[-1][:,2]) #Since every bird is its own neighbour
        for i in range(0, self.N-1):
            bin_ix = int(self.vector[-1][i,3])
            bin_iy = int(self.vector[-1][i,4])

            if bin_ix<1 or bin_ix > (self.Nbins-2):
                cond_x = np.logical_or(self.vector[-1][i + 1:, 3] > (bin_ix - 2) % self.Nbins,
                                       self.vector[-1][i + 1:, 3] < (bin_ix + 2) % self.Nbins)
            else:
                cond_x = np.logical_and(self.vector[-1][i + 1:, 3] > (bin_ix - 2),
                                        self.vector[-1][i + 1:, 3] < (bin_ix + 2))
            # print("and condx is",cond_x)
            if bin_iy < 1 or bin_iy > (self.Nbins-2):
                cond_y = np.logical_or(self.vector[-1][i + 1:, 4] > (bin_iy - 2) % self.Nbins,
                                       self.vector[-1][i + 1:, 4] < (bin_iy + 2) % self.Nbins)
            else:
                cond_y = np.logical_and(self.vector[-1][i + 1:, 4] > (bin_iy - 2),
                                        self.vector[-1][i + 1:, 4] < (bin_iy + 2))

            cond = np.logical_and(cond_x,cond_y)

            neigh_bins = i+1+np.argwhere(cond)

            distance_sq = (np.remainder(self.vector[-1][i,0] - self.vector[-1][neigh_bins,0] + self.L / 2., self.L) - self.L / 2.) ** 2 \
                          + (np.remainder(self.vector[-1][i,0] - self.vector[-1][neigh_bins,0] + self.L / 2., self.L) - self.L / 2.) ** 2
            indices = np.argwhere(distance_sq < self.R ** 2)
            indices = neigh_bins[indices] ##THis is here because distance_sq is an array with only the distance of the
            # birds in the neighbouring bins, so when you ask where distance<R it gives you the indices of this neighbouring bins
            #, so we need to go back to the actual indices of self.vector with neigh_bins
            Neighbours[i,indices] = self.vector[-1][i,2]
            Neighbours[indices,i] = self.vector[-1][indices,2]

        n = np.count_nonzero(~np.isnan(Neighbours),axis=0)
        avg_theta_r_cos = np.nansum(np.cos(Neighbours), axis=0) / n
        avg_theta_r_sin = np.nansum(np.sin(Neighbours), axis=0) / n
        theta_avg_r = np.arctan2(avg_theta_r_sin, avg_theta_r_cos) ### WARNING arctan2 gives 0 or pi in the case of two exactly opposite thetas
        self.vector[-1][:,2] = theta_avg_r + np.random.uniform(-self.eta / 2, self.eta / 2, size=np.size(theta_avg_r))


    def order_parameter_calculation(self):
        # Absolut value of the average normslized velocity is the order parameter of the system and checking its behaviour determines the phase transition
        v_a = np.sqrt((np.sum(np.cos(self.vector[:,:,2]),axis=1))**2 + (np.sum(np.sin(self.vector[:,:,2]),axis=1))**2)/self.N
        # v_a is an array with Nsteps components, each component is the order parameter for each step
        # Also we need the mean value of v_a after transient phase:
        r = v_a[v_a > 0.5]  # Eliminating values of order parameters in the transient regime, we assumed if order parameter is more than 0.5 then teh system has faced the phase transition
        if len(r) != 0:
            mean_v_a = np.sum(r) / len(r)  # Mean value of order parameter for each set of initial condition
        else:
            mean_v_a = 0
            print('Mean v_a does not reach 0.5')
        return v_a, mean_v_a  # TODO check what happens when no va are above 0.5
        #TODO plot v_a as function of rho and eta to see the phase transition

    def update(self):
        for i in range(self.Nsteps):  #i is the loop variable of the timestep, hard capped at 10 for now
            # if i%10 == 0: print(i)
            self.evolve()
            self.new_theta()

        #TODO Lastly, check if the dispersion of average theta is close to the noise ;
        self.va = self.order_parameter_calculation()
        #if dispersion in range()
        #Timestep = i
        #print(Timestep)
        #break


if __name__ == '__main__':
    seed = 1
    vel = 0.033
    N = 100
    R = 1
    L = 50
    eta = 0.3
    dt = 1
    Nsteps = 300
    # Run simulation

    RUN = True
    ANIMATE = True
    SAVE = False
    READ = False

    start_time = time.time()
    if RUN:
        swarm = Bird(seed,vel,N,R,L,eta,dt,Nsteps)
    print("--- %s seconds ---" % (time.time() - start_time))
    # Plots and animations:
    # Set to True to animate swarm motion

    if SAVE:
        # np.savetxt("Vector1.csv", swarm.vector, delimiter=",")
        name = input('Desired file name identification?\t')
        with open('Output_files/' + name + '_pickle_swarm.csv', 'wb') as pickle_out:
            pickle.dump(swarm, pickle_out)

    if READ:
        name = input('Desired file name identification?\t')
        with open('Output_files/' + name + '_pickle_swarm.csv', "rb") as pickle_in:  # "rb" because we want to read in binary mode
            swarm = pickle.load(pickle_in)

    if ANIMATE:
        def make_step(i):  # New animation with arrows and color, for old dot-only check version of git prior to  nov 28
            ax.clear()  # Clear the previous frame
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)

            # Calculate the color based on the orientation (theta)
            colors = np.pi + np.arctan2(np.sin(swarm.vector[i][:, 2]), np.cos(swarm.vector[i][:, 2]))

            # Use quiver to plot arrows with color based on orientation
            arrows = ax.quiver(
                swarm.vector[i][:, 0],  # x-coordinates
                swarm.vector[i][:, 1],  # y-coordinates
                np.cos(swarm.vector[i][:, 2]),  # cos(theta) as x-component of arrow
                np.sin(swarm.vector[i][:, 2]),  # sin(theta) as y-component of arrow
                scale = 1,
                width = R/L/2,
                scale_units = 'xy',  # use the same scale for x and y directions
                color = cm.twilight(colors / (2 * np.pi)),
                # Here we use the circular color map twilight to assign a color to the bird's directions
                headwidth = 5,
                headlength = 5  # width and length of the arrowhead
            )

            return arrows


        fig, ax = plt.subplots()
        bird_animation = animation.FuncAnimation(fig, make_step, frames=Nsteps, interval=1, blit=False)
        plt.show()



    # if ANIMATE:
    #     def make_step(i):
    #         # swarm.step(dt)
    #         # line = ax.quiver(swarm.vector[i][:,0:1],np.cos(swarm.vector[i][:,2]),np.sin(swarm.vector[i][:,2]))
    #         line.set_data(swarm.vector[i][:, 0], swarm.vector[i][:, 1])
    #         return line
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, aspect='equal')
    #     line, = ax.plot([], [],'o', ms=R/L*100)
    #     ax.set_xlim(0, L)
    #     ax.set_ylim(0, L)
    #     bird_animation = animation.FuncAnimation(fig, make_step, frames=Nsteps, interval=50, blit=False)
    #     plt.show()

    # Create input geometry from TOML file
    # inp = geometry('values.toml')

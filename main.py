# -*- coding: utf-8 -*-

# import csv
import numpy as np

import matplotlib; matplotlib.use("TkAgg")
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import time
import requests

# from Parameter_variation import *

# import pickle

    # =============================================================================
    #  PHY571 Numerical Physics Project
    # =============================================================================
    # ============================Start of preamble================================
    # This Python-function belongs to the PHY571 Numerical Physics Project. The file 
    # contains a function that calculates the dynamics of a swarm of interacting 
    # boids specified by the user
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
        vector = np.hstack((posvector, theta))
        vector = np.hstack((vector, np.zeros_like(theta)))
        vector = np.hstack((vector, np.zeros_like(theta)))
        vector = vector[np.newaxis,:, :]
        return vector

    """NOTATION: VECTOR[STEP][NBIRD,XYZBxBy]"""
    # the columns go as follows: X, Y , THETA (angle of boid with respect to horizontal), BIN (bin coordinate of boids position (bx,by))
    
    def __init__(self,seed,vel,N,R,L,eta,dt,Nsteps):
        self.vector = self.init_vector(int(seed),int(N),L)   # This is the NStepsx3 array that stores x,y and theta value at each step
                                                             # At each step, add another layer to the array (in 3D)
        self.velocity = vel                                  # Constant norm of velocity for all birds
        self.R = R                                           # Radius of influence of each boid
        self.L = L                                           # Size of the world
        self.eta = eta                                       # Interval of noise in theta
        self.dt = dt                                         # Constant time step
        self.N = N                                           # Number of particles
        self.Nsteps = Nsteps                                 # Number of timesteps
        self.Lbins = L/int(L/R)                              # Length of each bin

        #self.rho = self.N/(self.L)**2
        self.Nbins = int(self.L/self.Lbins) #There are Nbins**2. But each bird has a x and y coordinate bin: their
                                            # number ranging from 0 to Nbin-1
        self.update()                       # Runs code step after step and updates self.vector

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

    def new_theta(self):
        # Function to calculate the average angle of neighbouring boids and update
        # This function will be explained with the use of an example case of L = 10, R = 1.
        Neighbours = np.full((self.N, self.N), np.nan) # Initialise a NxN matrix, that shall keep track of which bird is
        # neighbour of which other bird. If bird number 1 is neighbour to bird 2, in the corresponding matrix element
        # (Neighbour[1,2]) the angle of bird 1 is introduced and in Neighbour[2,1] the angle of bird 2 is introduced.
        np.fill_diagonal(Neighbours, self.vector[-1][:,2]) #Since every bird is its own neighbour
        for i in range(0, self.N-1):
            bin_ix = int(self.vector[-1][i,3]) # x-bin-coordinate of the ith boid
            bin_iy = int(self.vector[-1][i,4]) # y-bin-coordinate of the ith boid

            # Throughout this function will compute the neighbouring birds of the ith bird, and fill the Neighbour matrix
            # step by step. But please note, once the neighbours of bird 1 are known, bird 2 does not need to consider
            # bird 1 in its calculation anymore, as it is already known whether they are neighbours of each other.
            # This way for bird i, we will only look at birds i+1. And the last bird (N) will already know all its neighbours.

            if bin_ix<1 or bin_ix > (self.Nbins-2): # in case the ith bird is in xbin 0 or 9 (out of 10-xbins) we
                # need to consider the boids in binx [9,0,1] or [8,9,0] respectively. This comes from transparent BC.
                cond_x = np.logical_or(self.vector[-1][i + 1:, 3] > (bin_ix - 2) % self.Nbins,
                                       self.vector[-1][i + 1:, 3] < (bin_ix + 2) % self.Nbins)
                # Therefore in case of bin_ix = 0, the bins to be taken are the ones fullfilling the criteria of cond_x,
                # which takes the neighbours either in a bin higher than 8 (so 9) 0R in the bins lower than 2 (so 0 and 1).
            else: # in case the ith bird is in binx between 1 and 8, the neighbours are those simply in [bin_ix-1,bin_ix,bin_ix+1]
                cond_x = np.logical_and(self.vector[-1][i + 1:, 3] > (bin_ix - 2),
                                        self.vector[-1][i + 1:, 3] < (bin_ix + 2))
                # Therefore the condition that they need to fullfill is that their bin coordinate is lower AND higher
                # than the max and min limits.
            # Equivalently for the y coordinate
            if bin_iy < 1 or bin_iy > (self.Nbins-2):
                cond_y = np.logical_or(self.vector[-1][i + 1:, 4] > (bin_iy - 2) % self.Nbins,
                                       self.vector[-1][i + 1:, 4] < (bin_iy + 2) % self.Nbins)
            else:
                cond_y = np.logical_and(self.vector[-1][i + 1:, 4] > (bin_iy - 2),
                                        self.vector[-1][i + 1:, 4] < (bin_iy + 2))

            cond = np.logical_and(cond_x,cond_y) #It should fullfill both the x and y conditions.

            neigh_bins = i+1+np.argwhere(cond) # The indices of the birds in the neighbouring bins are those that
            # fulfill the conditions (x and y)

            # Now for the birds inside neigh_bins (within a range of +-1 bins) of boid i, their distance with boid i
            # will be computed and compared to the radius of influence R.
            distance_sq = (np.remainder(self.vector[-1][i,0] - self.vector[-1][neigh_bins,0] + self.L / 2., self.L) - self.L / 2.) ** 2 \
                          + (np.remainder(self.vector[-1][i,1] - self.vector[-1][neigh_bins,1] + self.L / 2., self.L) - self.L / 2.) ** 2
            indices = np.argwhere(distance_sq < self.R ** 2) # This are the indices of self.vector[-1][neigh_bins,0]
            # that are within R of bird i, which means that these indices are indicies of neigh_bins and not of self.vector
            indices = neigh_bins[indices] # We need to map back the indices from self.vector[-1][neigh_bins,0] to self.vector[-1][:,0]
            Neighbours[i,indices] = self.vector[-1][i,2] # The birds indices all have neighbour i
            Neighbours[indices,i] = self.vector[-1][indices,2] # The bird i has neighbours indices

        n = np.count_nonzero(~np.isnan(Neighbours),axis=0)
        avg_theta_r_cos = np.nansum(np.cos(Neighbours), axis=0) / n
        avg_theta_r_sin = np.nansum(np.sin(Neighbours), axis=0) / n
        theta_avg_r = np.arctan2(avg_theta_r_sin, avg_theta_r_cos) ### WARNING arctan2 gives 0 (or pi) in the case of two exactly opposite thetas
        self.vector[-1][:,2] = theta_avg_r + np.random.uniform(-self.eta / 2, self.eta / 2, size=np.size(theta_avg_r))


    def order_parameter_calculation(self):
        # Absolut value of the average normslized velocity is the order parameter of the system and checking its behaviour determines the phase transition
        v_a = np.sqrt((np.sum(np.cos(self.vector[:,:,2]),axis=1))**2 + (np.sum(np.sin(self.vector[:,:,2]),axis=1))**2)/self.N
        # v_a is an array with Nsteps components, each component is the order parameter for each time step
        # Also we need the mean value of v_a for each set of initial parameters
        mean_v_a = np.mean(v_a[-51:-1])
        print('Mean v_a (last 50) is', mean_v_a)
        # plt.plot(v_a) #Uncomment this line in case one would like to see va over timestep
        # plt.show()
        return v_a, mean_v_a


    def update(self):
        for i in range(self.Nsteps):  #i is the loop variable of the timestep, hard capped at 10 for now
            # if i%10 == 0: print(i)
            self.evolve()
            self.new_theta()

        self.va , self.mean_va = self.order_parameter_calculation()

if __name__ == '__main__':
    seed = 1
    vel = 0.033
    N = 100
    R = 1
    L = 10
    eta = 1
    dt = 1
    Nsteps = 100
    # Run simulation

    RUN = True
    ANIMATE = True #Animate dynamics of particles
    GIF = False # Save animation in gif?

    SAVE = False
    READ = False

    start_time = time.time()
    if RUN:
        swarm = Bird(seed,vel,N,R,L,eta,dt,Nsteps)
    print("--- %s seconds ---" % (time.time() - start_time))


    if SAVE:
        name = input('Desired file name identification?\t')
        with open('New_output_files/' + name + '_pickle_swarm.csv', 'wb') as pickle_out:
            pickle.dump(swarm, pickle_out)

    if READ:
        name = input('Desired file name identification?\t')
        with open('New_Output_files/' + name + '_pickle_swarm.csv', "rb") as pickle_in:  # "rb" because we want to read in binary mode
            swarm = pickle.load(pickle_in)

    # Plots and animations:
    # Set to True to animate swarm motion
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
                scale = 2, #1 #Set to 1 in case of L large (more than 20), and to 2 in case of L small
                width = 1/7/50,#R/L/50 # use the second width R/L/2 if one would like to scale the size of the arrows with their size relative to the box
                scale_units = 'xy',  # use the same scale for x and y directions
                color = cm.twilight(colors / (2 * np.pi)),
                # Here we use the circular color map twilight to assign a color to the bird's directions
                headwidth = 5,
                headlength = 5  # width and length of the arrowhead
            )

            return arrows


        fig, ax = plt.subplots()
        bird_animation = animation.FuncAnimation(fig, make_step, frames=Nsteps, interval=5, blit=False)
        if GIF:
            gif_filename = input('Desired GIF file name identification?\t')
            writer = animation.PillowWriter(fps=20)
            ## Save the animation as a GIF
            bird_animation.save('Animations/'+gif_filename+'.gif', writer=writer)
        plt.show()

    # Create input geometry from TOML file
    # inp = geometry('values.toml')

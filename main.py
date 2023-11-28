# -*- coding: utf-8 -*-

# import csv
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time


# from Parameter_variation import *

# This is the branch before optimisation

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
        vector = np.hstack((posvector, theta))
        vector = vector[np.newaxis,:, :]
        return vector


    """NOTATION: VECTOR[STEP][NBIRD,XYZ]"""
    
    def __init__(self,seed,vel,N,R,L,eta,dt,Nsteps):
        self.vector = self.init_vector(seed,N,L)   # This is the NStepsx3 array that stores x,y and theta value at each step
                                                 #At each step, add another layer to the array (in 3D)
        self.velocity = vel                      # Constant norm of velocity for all birds
        self.R = R                               # Radius of influence of each boid
        self.L = L                               # Size of the world
        self.eta = eta                           # Interval of noise in theta
        self.dt = dt                             # Constant time step
        self.N = N
        self.Nsteps = Nsteps
        #self.rho = self.N/(self.L)**2

        self.update()


    def evolve(self):
        # Function evolves the movement of the boids for one timestep dt and velocity v
        vector_old = self.vector[-1]
        vector_add = np.zeros_like(vector_old)
        theta = vector_old[:, 2]
        vector_add[:,0] = self.velocity*self.dt*np.cos(theta)
        vector_add[:,1] = self.velocity*self.dt*np.sin(theta)
        vector_new = vector_old + vector_add

        vector_new[:,0:2] = vector_new[:,0:2] % self.L # Periodic boundary conditoins

        vector_new_reshaped = vector_new.reshape(1, self.N, 3) #reshape the new post matrix in order to be able to concatenate
        self.vector = np.concatenate((self.vector, vector_new_reshaped), axis=0)


        # vector_new = vector_old[:,0:1] +

    def new_theta(self):
        # Function to calculate the average angle of neighbouring boids and update
        Neighbours = np.full((self.N, self.N), np.nan)
        np.fill_diagonal(Neighbours, self.vector[-1][:,2]) #Since every bird is its own neighbour
        for i in range(0, self.N):
            distance_sq = (self.vector[-1][i,0] - self.vector[-1][i+1:,0]) ** 2 + (self.vector[-1][i,1] - self.vector[-1][i+1:,1]) ** 2
            indices = np.argwhere(distance_sq < self.R ** 2)
            Neighbours[i,i+1+indices] = self.vector[-1][i+1+indices,2]
            Neighbours[i+1+indices,i] = self.vector[-1][i,2]

        # print(Neighbours)
        n = np.count_nonzero(~np.isnan(Neighbours),axis=1)
        # n = np.count_nonzero(Neighbours, axis=1)
        avg_theta_r_cos = np.nansum(np.cos(Neighbours), axis=1) / n
        avg_theta_r_sin = np.nansum(np.sin(Neighbours), axis=1) / n
        theta_avg_r = np.arctan2(avg_theta_r_sin, avg_theta_r_cos) ### WARNING arctan2 gives 0 or pi in the case of two exactly opposite thetas
        # TODO make sure there are no thetas going over 2pi and under 0
        self.vector[-1][:,2] = theta_avg_r + np.random.uniform(-self.eta / 2, self.eta / 2, size=np.size(theta_avg_r))


    # def check_transition(self): # we are not sure if this is a good idea, nevertheless we can plot it
    #
    #     last_thetas = self.vector[-1][:][2]
    #     avg_theta_r_cos = np.matrix.sum(np.cos(last_thetas), axis=1) / self.N
    #     avg_theta_r_sin = np.matrix.sum(np.sin(last_thetas), axis=1) / self.N
    #     theta_avg_last = np.atan2(avg_theta_r_sin, avg_theta_r_cos)
    #
    #     second_to_last_thetas = self.vector[-2][:][2]
    #     avg_theta_r_cos = np.matrix.sum(np.cos(second_to_last_thetas), axis=1) / self.N
    #     avg_theta_r_sin = np.matrix.sum(np.sin(second_to_last_thetas), axis=1) / self.N
    #     theta_avg_second_to_last = np.atan2(avg_theta_r_sin, avg_theta_r_cos)
    #
    #     diff = np.min(np.abs(theta_avg_last-theta_avg_second_to_last),np.abs(theta_avg_last-theta_avg_second_to_last-2*np.pi))

    def order_parameter_calculation(self):
        # Absolut value of the average normslized velocity is the order parameter of the system and checking its behaviour determines the phase transition
        v_a = np.sqrt((np.sum(np.cos(self.vector[:, :, 2]), axis=1)) ** 2 + (
            np.sum(np.sin(self.vector[:, :, 2]), axis=1)) ** 2) / self.N
        # v_a is an array with Nsteps components, each component is the order parameter for each step
        # Also we need the mean value of v_a after transient phase
        r = v_a[v_a > 0.5]  # Eliminating values of order parameters in the transient regime, we assumed if order parameter is more than 0.5 then teh system has faced the phase transition
        # mean_v_a = np.sum(r) / len(r)
        mean_v_a = np.mean(r) # Mean value of order parameter for each set of initial condition
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
    N = 1000
    R = 0.1
    L = 10
    eta = 0.1
    dt = 1
    Nruns = 1000
    # Run simulation

    RUN = True
    ANIMATE = False
    SAVE = False
    READ = False

    start_time = time.time()
    if RUN:
        swarm = Bird(seed,vel,N,R,L,eta,dt,Nruns)
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
        def make_step(i):
            # swarm.step(dt)
            # line = ax.quiver(swarm.vector[i][:,0:1],np.cos(swarm.vector[i][:,2]),np.sin(swarm.vector[i][:,2]))
            line.set_data(swarm.vector[i][:, 0], swarm.vector[i][:, 1])
            return line

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        line, = ax.plot([], [], 'bo', ms=R/L*100)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        bird_animation = animation.FuncAnimation(fig, make_step, frames=Nruns, interval=1, blit=False)
        plt.show()
    # Create input geometry from TOML file
    # inp = geometry('values.toml')

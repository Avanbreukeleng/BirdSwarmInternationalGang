# -*- coding: utf-8 -*-

# import csv
import numpy as np
import pickle
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

    def init_vector(self, seed, N):
        np.random.seed(seed)
        posvector = np.random.rand(N, 2)
        theta = np.random.uniform(0, 2*np.pi,N)
        vector = np.hstack((posvector, theta))
        return vector
    
    def __init__(self,seed,vel,N,R,L,eta,dt):
        self.vector = self.init_vector(seed,N)   # This is the NStepsx3 array that stores x,y and theta value at each step
                                                 #At each step, add another layer to the array (in 3D)
        self.velocity = vel                      # Constant norm of velocity for all birds
        self.R = R                               # Radius of influence of each boid
        self.L = L                               # Size of the world
        self.eta = eta                           # Interval of noise in theta
        self.dt = dt                             # Constant time step
        self.N = N

        self.update()


if __name__ == '__main__':
    # Set to True to show input validation plots
    # PLOT = True

    # Create input geometry from TOML file
    # inp = geometry('values.toml')

    # Run simulation
    output = Swarm(inp)


    # def Swarm(inp):
    #
    # return output
# -*- coding: utf-8 -*-

# import csv
import numpy as np
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



    def evolve(self):
        # Function evolves the movement of the boids for one timestep dt and velocity v
        vector_old = self.vector[-1]
        vector_add = np.zeros_like(vector_old)
        theta = vector_old[:, 2]
        vector_add[:,0] = self.velocity*self.dt*np.cos(theta)
        vector_add[:,1] = self.velocity*self.dt*np.sin(theta)
        vector_new = vector_old + vector_add

        for j in range(0, self.N) #can probably be improved using x%L and y%L
            x = vector_new[j,0]
            y = vector_new[j,1]
            if x > L:
                vector_new[j,0] = x - L
            if x < 0:
                vector_new[j,0] = x + L
            if y > L:
                vector_new[j,1] = y - L
            if y < 0:
                vector_new[j,1] = y + L
        vector_new.reshaped = vector_new.reshape(1, self.N, 3) #reshape the new post matrix in order to be able to concatenate
        self.vector = np.concatenate((self.vector, vector_new.reshaped), axis=0)


        vector_new = vector_old[:,0:1] +

    def new_theta(self):
        # Function to calculate the average angle of neighbouring boids and updates
        Neighbours = np.zeros(self.N, self.N)
        np.fill_diagonal(Neighbours, self.vector[-1][:][2]) #Since every bird is its own neighbour
        for i in range(0, self.N):
            # Sigma_theta = self.vector[-1][i][2]  # Sum of neighbours' thetas
            for j in range(i + 1, self.N):
                if (self.vector[-1][i][0] - self.vector[-1][j][0]) ** 2 + (
                        self.vector[-1][i][1] - self.vector[-1][j][1]) ** 2 <= self.R ** 2:
                    Neighbours[i][j] = self.vector[-1][i][2]
                    Neighbours[j][i] = self.vector[-1][j][2]

        n = np.count_nonzero(Neighbours, axis=1)
        avg_theta_r_cos = np.matrix.sum(np.cos(Neighbours), axis=1) / n
        avg_theta_r_sin = np.matrix.sum(np.sin(Neighbours), axis=1) / n
        theta_avg_r = np.atan2(avg_theta_r_sin, avg_theta_r_cos) ### WARNING arctan2 gives 0 or pi in the case of two exactly opposite thetas
        # TODO make sure there are no thetas going over 2pi and under 0
        self.vector[-1][:][2] = theta_avg_r + np.random.uniform(-self.eta / 2, self.eta / 2, size=np.size(theta_avg_r))


    # def check_transition(self): # we are not sure if this is a good idea, nevertheles we can plot it
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



    def update(self):
    for i in range(0, 100) #i is the loop variable of the timestep, hard capped at 100 for now
        self.evolve()
        self.new_theta()
    #TODO Lastly, check if the dispersion of average theta is close to the noise ;
        #if dispersion in range()
        #Timestep = i
        #print(Timestep)
        #break

def make_step(i):
    # swarm.step(dt)
    line.set_data(swarm.vector[i][:][0], swarm.vector[i][:][1])
    return line

def animate(vector):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    line = ax.plot([], [], 'bo', ms=25)
    # TODO understand comma?
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    anim = animation.FuncAnimation(fig, make_step(i), interval=1, blit=False)

def Plot():



anim = animation.FuncAnimation(fig, animate, interval=1, blit=False)

if __name__ == '__main__':
    seed = 1
    vel = 0.033
    N = 1
    R = 1
    L = 10
    eta = np.pi/4
    dt = 1
    runs = 10
    # Run simulation
    swarm = Bird(seed,vel,N,R,L,eta,dt,runs)

    # Plots and animations:
    # Set to True to animate swarm motion
    ANIMATE = True

    if ANIMATE:
        animate(swarm.vector)

    # Create input geometry from TOML file
    # inp = geometry('values.toml')

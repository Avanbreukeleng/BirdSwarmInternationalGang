import pickle
import re
import numpy as np
import csv
import pandas as pd
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# from objects.mesh import Mesh
# from objects.element import Element


class Assembly:
    def __init__(self, input, show=False) -> None:

        # Read user input dictionary
        self.input = input

        # Initialise output
        self.output = {'name': self.input['name']}

    def plot_input(self, show=True, save=False) -> None:

        # Check the provided input
        # --------------------------------------------------------------------------
        # if 'parts' in self.input.keys() is False:
        #     raise ('Error: self.input.parts has not been specified by user')

        assert 'parts' in self.input.keys(), \
            '\n \nError: self.input.parts has not been specified by user'

        # Display the structure for the user to check it visually
        # --------------------------------------------------------------------------
        # Plot the parts
        for ix in range(len(self.input['parts'][0, :])):
            plt.plot(self.input['points'][0, self.input['parts'][:, ix] - 1],
                     self.input['points'][1, self.input['parts'][:, ix] - 1], 'k-', marker='o', linewidth=2,
                     markersize=12)

        # Plot the points
        plt.plot(self.input['points'][0, :], self.input['points']
        [1, :], 'or', linewidth=4, markersize=10)

        # Format the axes of the plot
        plt.xlim([min(self.input['points'][0, :]) - 0.5, max(self.input['points'][0, :]) + 0.5])
        plt.ylim([min(self.input['points'][1, :]) - 0.5, max(self.input['points'][1, :]) + 0.5])
        plt.rcParams.update({'font.size': 20})

        plt.title("Input (assembly) Plot")

        # Optionally, save and show figure
        if save:
            plt.savefig('mesh_1.png')
        if show:
            plt.show()
            # plt.savefig('Input (assembly) Plot.pdf')

    def plot_output(self, new_coordinates=None):

        # Plot output
        # TODO: improve plots to ease comparison, etc.

        if np.all(self.input['type'] == 1):
            # Reshape 3 displacements per node
            locDisp3D = self.output['U'].reshape(3, self.mesh['nNodes'], order='F')
        else:
            # Reshape 2 displacements per node
            locDisp3D = self.output['U'].reshape(2, self.mesh['nNodes'], order='F')

        # 2D locations of the displaced nodes
        locDisp = self.mesh['nodes'] + self.input['plot_scale'] * locDisp3D[0:2, :]

        # Plot elements (lines linking nodes)
        plt.plot(self.input['points'][0, self.input['parts'][:, 0] - 1],
                 self.input['points'][1, self.input['parts'][:, 0] - 1], 'gray',
                 linewidth=2, alpha=0.5, label="Original Structure")

        for ix in range(1, max(self.input['parts'][0, :].shape)):
            plt.plot(self.input['points'][0, self.input['parts'][:, ix] - 1],
                     self.input['points'][1, self.input['parts'][:, ix] - 1], 'gray',
                     linewidth=2, alpha=0.5)

        # Plot NEW elements (lines linking nodes)
        # plt.plot(self.output['mesh']['part'][0]['new_nodes'][0],
        #          self.output['mesh']['part'][0]['new_nodes'][1],
        #          'lightcoral', linewidth=2, alpha=0.5, label = "Deformed Structure")

        # cmap = ListedColormap(['r','g','b'])
        # norm
        stresses = self.output['stressmax']
        strains = self.output['strainmax']

        for ix in range(0, len(self.output['mesh']['part'])):
            # print(len(self.output['mesh']['part']['ix']['nNodes'])-1)
            for j in range(self.output['mesh']['part'][ix]['nNodes'] - 1):
                stress = self.output['mesh'].mesh['part'][ix]['stress'][j]
                rgb = [1, 1, 1]
                if stress >= 0:
                    rgb[0] = 1 - (stress / stresses[1])
                    rgb[1] = 1 - (stress / stresses[1])
                else:
                    rgb[1] = 1 - (stress / stresses[0])
                    rgb[2] = 1 - (stress / stresses[0])

                plt.plot(self.output['mesh'].mesh['part'][ix]['new_nodes'][0][j:j + 2],
                         self.output['mesh'].mesh['part'][ix]['new_nodes'][1][j:j + 2], linewidth=4, alpha=0.8,
                         color=rgb)

        # Plot original nodes
        plt.plot(self.mesh['nodes'][0, :], self.mesh['nodes'][1, :], 'o',
                 linewidth=2, markerfacecolor='None', markersize=0)

        # Plot displaced nodes
        plt.plot(locDisp[0, :], locDisp[1, :], 'or', linewidth=2,
                 markerfacecolor='b', markersize=0)

        # Set axis limits
        plt.grid(alpha=0.5)
        plt.legend(fontsize="x-small", loc="upper center")
        plt.xlim([
            min(min(self.input['points'][0, :]), min(locDisp[1, :])) - 50,
            max(self.input['points'][0, :]) + 50
        ])
        plt.ylim([
            min(self.input['points'][1, :]) - 0.5,
            max(max(self.input['points'][1, :]), max(locDisp[1, :])) + 0.5
        ])
        plt.title("Output Plot")
        plt.show()

    def save_output(self):
        # TODO: export output for analysis
        # with open('output.csv', 'w') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(self.output.keys())
        #     writer.writerow(self.output.values())

        save = bool(input('Do you want to save the results? False = enter, True = type something \t'))
        if save is True:
            name = input('Desired file name identification?\t')
            if all(self.input['type']) == 1:
                with open(name + 'pickle_output_beam.csv', 'wb') as pickle_out:
                    pickle.dump(self.output, pickle_out)
            elif all(self.input['type']) == 0:
                with open(name + '_pickle_output_truss.csv', 'wb') as pickle_out:
                    pickle.dump(self.output, pickle_out)

        # ACCESSING PICKLED OUTPUT DATA
        with open('TestingFilepickle_output.csv', 'rb') as pickle_in:
            data = pickle.load(pickle_in)
            print(data)
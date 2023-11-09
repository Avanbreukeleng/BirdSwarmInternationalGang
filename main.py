# -*- coding: utf-8 -*-

# import csv
import numpy as np
import pickle
# from objects.assembly import Assembly
# from objects.mesh import Mesh
# from objects.element import Element

from structure import geometry


def Swarm(inp):
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

    # --------------------------------------------------------------------------
    # 0. User Input
    # --------------------------------------------------------------------------
    assembly = Assembly(inp)

    # --------------------------------------------------------------------------
    # 1. Input Check
    # --------------------------------------------------------------------------
    if PLOT:
        # pass
        # UNCOMMENT THE LINE BELOW and remove 'pass' above
        assembly.plot_input()

    # --------------------------------------------------------------------------
    # 2. Creation of Parts - Plot for check by user
    # --------------------------------------------------------------------------
    # Create mesh
    assembly.mesh = Mesh(assembly)
    # print(assembly.getmembers())
    # print(vars(assembly))

    # name = input('Desired file name identification?\t')
    # with open(name + 'pickle_assembly.csv', 'wb') as pickle_out:
    #     pickle.dump(assembly, pickle_out)
    #
    if PLOT:
        # Plot mesh

        # UNCOMMENT THE LINE BELOW
        assembly.mesh.plot_mesh()

        # Show the user the mesh overlaid over the geometry
        assembly.mesh.plot_mesh(False)

        # UNCOMMENT THE LINE BELOW
        assembly.plot_input()

        # Pause for user to do visual inspection of structure and mesh
        input('Please inspect the mesh shown in figures 1 and 2 and 3. \
            They may also be saved in the working path as "mesh_1.png", "mesh_2.png", and "mesh_3.png", if so configured. \
            Press enter to continue.')

    # --------------------------------------------------------------------------
    # 3. Creation of boids
    # --------------------------------------------------------------------------
    # Assign properties
    assembly.mesh.assign_element_properties()

    # Creation of Elements
    for ix in range(assembly.mesh.mesh['nElements']):
        # for each element, generate local properties
        element = Element(assembly, ix)
        # rewrite the local mesh with the mesh that now has the element properties included
        assembly.mesh = element.assembly.mesh



    # --------------------------------------------------------------------------
    # 4. Performing the simulation
    # TODO: create step function

    # --------------------------------------------------------------------------
    o = assembly.output
    # print('Assembly output content\n', o.keys())
    # print(o['inactiveDF'])


    # --------------------------------------------------------------------------
    # 5. Plot results
    # TODO: improve plots to ease comparison, etc.

    # --------------------------------------------------------------------------

    # UNCOMMENT THE LINE BELOW
    assembly.plot_output()

    # --------------------------------------------------------------------------
    # 6. Save simulation results

    # TODO: fill the block (consider CSV, pickling)
    # --------------------------------------------------------------------------
    assembly.save_output()
    return assembly.output


if __name__ == '__main__':
    # Set to True to show input validation plots
    # PLOT = True

    # Create input geometry from TOML file
    inp = geometry('values.toml')

    # Run simulation
    output = Swarm(inp)

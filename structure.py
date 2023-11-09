import toml
import numpy as np

from materials import read_materials


# push and pull test

def create_simulation_input(
        # --------------------------------------------------------------------------
        name,
        # Geometry and discretisation
        r, R, v, eta, N, seed,
        # # Forces
        # forces,
        # Boundary conditons
        bc,
        #sim conditions
        sim
        # Output settings
        # nFreq, plot_scale
        # --------------------------------------------------------------------------
) -> dict:
    # Initialize output
    input = {}

    input['name'] = name

    input['boids'] = np.zeros((N,3)) # [x,y,theta] for N boids

    return input
#
#
def read_input(input_file) -> dict:
    d = toml.load(input_file)

    # Simulation name
    name = d['name']

    # Geometry
    geometry = d['Geometry']

    # Boundary conditions
    bc = d['BoundaryConditions']

    # Simulation conditions
    sim = d['SimulationConditions']

    # # Applied forces
    # forces = d['Forces']
    #
    # # Output settings
    # output_settings = d['Output']

    return name, geometry, bc, sim


def geometry(input_file):
    # Read input from TOML file
    name, geometry, bc, sim = read_input(input_file)

    # Call the function that creates the geometry
    input = create_simulation_input(
        name=name,
        # Geometry, discretisation
        **geometry,
        # forces=forces,
        # Boundary condition application
        bc=bc,
        # sim conditoins application
        sim=sim,
        # Output settings
        # **output_settings
    )
    return input
import sys
sys.path.insert(0, '/Users/jonathanzhang/Documents/ucsf/kortemme-pinney/repositories/kinetics_simulator')
import numpy as np
from collections import OrderedDict
from src import chemicalkinetics

def initialize_mass_action_test_rxn():
    """
    Initializes mass action reaction test case. Concentration and
    time units are in µM and s respectively.
    """
    reaction = {
    'P + B <-> C': OrderedDict({'kon': 10, 'koff': 10}),
    }
    initial_values = {'P': 100, 'B': 100}
    return reaction, initial_values

def initialize_michaelis_menten_test_rxn():
    """
    Initializes MM reaction test case. Concentration and
    time units are in µM and s respectively.
    """
    reaction = {
        'S + E -> E + P': OrderedDict({'kcat': 100, 'Km': 100})
    }
    initial_values = {'S': 1000, 'E': 0.01}
    return reaction, initial_values

def test_mass_action():
    tol = 10e-8
    time = np.linspace(0, 100, 100)
    reaction, initial_values = initialize_mass_action_test_rxn()
    reaction_network = chemicalkinetics.ChemicalReactionNetwork(reaction, {}, initial_values, time, 'µM', 's')

    # check that the dimensionality of reaction matrices is correct
    assert len(reaction_network.mass_action_reactions.reactions) == 2
    assert len(reaction_network.species) == 3
    assert len(reaction_network.mass_action_reactions.rate_names) == 2
    assert reaction_network.mass_action_reactions.K.shape == (2,2)
    assert reaction_network.mass_action_reactions.A.shape == (2,3)
    assert reaction_network.mass_action_reactions.N.shape == (2,3)

    # hand-calculated rates at t=0
    rates_t0 = {'P': -100000, 'B': -100000, 'C': 100000}

    # check that initial rates are computed correctly
    rates = reaction_network.ODEs(reaction_network.initial_concentrations, 0, reaction_network.michaelis_menten_reactions)
    for index, specie in enumerate(reaction_network.species):
        assert np.abs(rates_t0[specie] - rates[index]) < tol

    # check that equilibrium concentrations are computed correctly
    reaction_network.integrate(reaction_network.initial_concentrations, time)
    equilibrium_concens = reaction_network.concentrations[:, -1]
    assert np.abs(equilibrium_concens[reaction_network.species.index('C')] - (equilibrium_concens[reaction_network.species.index('P')] * equilibrium_concens[reaction_network.species.index('B')])) < tol

def test_michaelis_menten():
    tol = 10e-8
    time = np.linspace(0, 100, 100)
    reaction, initial_values = initialize_michaelis_menten_test_rxn()
    reaction_network = chemicalkinetics.ChemicalReactionNetwork({}, reaction, initial_values, time, 'µM', 's')

    # check that dimensionality is correct
    assert len(reaction_network.michaelis_menten_reactions.reactions) == 1
    assert len(reaction_network.michaelis_menten_reactions.substrate_indices) == 1 
    assert len(reaction_network.michaelis_menten_reactions.substrate_stoichiometries) == 1 
    assert len(reaction_network.michaelis_menten_reactions.product_indices) == 1
    assert len(reaction_network.michaelis_menten_reactions.product_stoichiometries) == 1
    assert len(reaction_network.michaelis_menten_reactions.enzyme_indices) == 1
    assert len(reaction_network.michaelis_menten_reactions.Km_names) == 1
    assert len(reaction_network.michaelis_menten_reactions.Kms) == 1
    assert len(reaction_network.michaelis_menten_reactions.kcat_names) == 1
    assert len(reaction_network.michaelis_menten_reactions.kcats) == 1

    # hand-calculated rates at different 0, 10, 100, 1000 µM [S]
    rates = [0, 0.09090157831934675, 0.4999874999995768, 0.9090901577678778, 0.9900990002279286]
    concentrations = [0, 10, 100, 1000, 10000]

    # check that initial velocities are computed properly
    species_dict = {'E': 0.01, 'P': 0}
    for rate, concentration in zip(rates, concentrations):
        species_dict['S'] = concentration
        v = reaction_network.michaelis_menten_reactions.compute_velocities(np.array([species_dict[i] for i in reaction_network.species]))
        assert abs(v[reaction_network.species.index('S')] + rate) < tol
        assert abs(v[reaction_network.species.index('P')] - rate) < tol
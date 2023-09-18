""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for modelling the kinetics of chemical reaction networks.
"""

#imports
import re
import numbers
import numpy as np
from scipy.integrate import odeint
from typing import Union
from scipy.optimize import curve_fit

def mm_to_mass_action():
    """ 
    Identifies a mass action kinetic model consistent
    with the provided Michaelis-Menten model.
    """
    pass

class SpeciesCollection:
    """ 
    Utility class for organizing chemical species concentration data.
    """

    def __init__(self, names: list):
        self.names = names
        self.values, self.lb, self.ub, self._mapper = None, None, None, None 

    def set_values_and_bounds(self, concentration_dict: dict):

        value_def, lb_def, ub_def = 0, 0, 1e3
        values = np.array([value_def if specie not in concentration_dict.keys() else concentration_dict[specie]['conc'] for specie in self.names])
        lb = np.array([lb_def if specie not in concentration_dict.keys() else concentration_dict[specie]['conc-lb'] for specie in self.names])
        ub = np.array([ub_def if specie not in concentration_dict.keys() else concentration_dict[specie]['conc-ub'] for specie in self.names])
        self.values, self.lb, self.ub = values, lb, ub
        self._mapper = dict([(name, index) for index, name in enumerate(self.names)])

    def get_specie_value(self, name: str):
        return self.values[self._mapper[name]]

    def get_specie_lb(self, name: str):
        return self.lb[self._mapper[name]] 

    def get_specie_ub(self, name: str):
        return self.ub[self._mapper[name]] 

class MassActionReactions:
    """ 
    Utility class organizing reaction stoichiometries and rate constants.

    Attributes
    ----------
    reactions: list
        A list of chemical equations represented as strings.
    A: np.ndarray
        The substrate stoichiometry matrix for the mass action reactions.
    N: np.ndarray
        The reaction stoichiometry matrix for the mass action reactions.
    rconst_names: list
        A list of rate names for each rate constant.
    rconst_values: list
        A list of rate constant values.
    rconst_lb: list
        A list of lower bounds for rate constants.
    rconst_ub: list
        A list of upper bounds for rate constants.
    K: np.ndarray
        The rate matrix for the mass action reactions.
    """

    def __init__(
            self, 
            reactions:list, 
            A: list, 
            N: list, 
            rconst_names: list, 
            rconst_values: list , 
            rconst_lb: list, 
            rconst_ub: list
            ):

        # attributes holding stoichiometry and reaction data
        self.reactions = reactions
        self.N = np.vstack(N)
        self.A = np.vstack(A)

        # attributes holding rate constant data
        self.rconst_names = rconst_names
        self.rconst_values = rconst_values
        self.rconst_lb = rconst_lb 
        self.rconst_ub = rconst_ub
        self.K = np.diag(np.array(rconst_values))
        self._mapper = dict([name, index] for index, name in enumerate(self.rconst_names))

    def get_rconst_value(self, name: str):
        return self.rconst_values[self._mapper[name]]

    def get_rconst_lb(self, name: str):
        return self.rconst_lb[self._mapper[name]] 

    def get_rconst_ub(self, name: str):
        return self.rconst_ub[self._mapper[name]]

class ChemicalReactionNetwork:
    """
    Class for an arbitrary network of chemical reactions.

    Attributes
    ----------
    species: list
        A list of all chemical species present.
    mass_action_reactions: list
        A list of MassActionReactions objects, each describing a
        reaction modelled by mass action kinetics.
    michaelis_menten_reactions: list
        A list of MichaelisMentenReactions objectseach 
        describing a reaction modelled by MM kinetics.
    time: np.ndarray
        An array of times to solve over.
    initial_concentrations: np.ndarray  
        Defines the initial state of the chemical system.
    update_dictionary: dict
        A dictionary of functions for updating reaction rates or 
        initial concentrations. 
    concentrations: np.ndarray
        Array of concentrations of each specie at each timepoint.
    """

    def __init__(self, concentration_units='µM', time_units='s'):

        self.concentration_units, self.time_units = concentration_units, time_units
        self.species_collection, self.mass_action_reactions = None, None
        
        self.time, self.timestep = None, None
        self.species_collection = None
        self.mass_action_reactions = None 
        self.update_dictionary = None 
        self.ODEs = None
        self.simulated_data = None 

    def from_dict(self, reaction_dict: dict):
        """ 
        Method for updating object with reaction network information from an 
        input dictionary.
        """

        # parse the reaction_dict to ensure function assumptions are satisfied
        assert type(reaction_dict) == dict, 'ChemicalReactionNetwork Error: reaction_dict must be a dictionary.'
        assert len(reaction_dict) > 0, 'ChemicalReactionNetwork Error: reaction_dict cannot be empty.'
        for key, value in reaction_dict.items():
            assert isinstance(key, str), 'ChemicalReactionNetwork Error: keys within reaction_dict must be string representations of a chemical reaction.'
            assert isinstance(value, dict), 'ChemicalReactionNetwork Error: values within reaction_dict must be dictionaries.'
            assert set(value.keys()) == set(['model', 'rconst-names', 'rconst-values', 'rconst-lb', 'rconst-ub']), 'ChemicalReactionNetwork Error: Reaction keys must be mapped to a dictionary containing "model", "rconst-names", "rconst-values", "rconst-lb", and "rconst-ub" keys.'
            assert value['model'] in set(['mass-action', 'michaelis-menten']), 'ChemicalReactionNetwork Error: "model" can only take on "mass-action" or "michaelis-menten".'
            assert isinstance(value['rconst-names'], (list, np.ndarray, str)) and isinstance(value['rconst-values'], (list, np.ndarray, numbers.Number)), 'ChemicalReactionNetworkError: Bidirectional reactions must define rate constant information as arrays. Otherwise, rate constant information can be numeric or string .'

        # update attributes
        characters = {'*', '->', '+', '<->'}
        species = list(set([specie for specie in sum([i.split(' ') for i in reaction_dict.keys()], []) if specie not in characters and not specie.isnumeric()]))
        self.species_collection = SpeciesCollection(names = species)
        updated_reaction_dict = ChemicalReactionNetwork._split_reversible_reactions(reaction_dict)
        self.mass_action_reactions = self._parse_reaction_dict(updated_reaction_dict)
        self.ODEs = self._define_ODEs()

    @staticmethod
    def _split_reversible_reactions(reaction_dict: dict):
        """
        Splits reversible reactions into elementary reactions.
        """
        updated_reaction_dict = {}
        for reaction in reaction_dict.keys():

            if 'fit' not in reaction_dict[reaction].keys():
                reaction_dict[reaction]['fit'] = False

            if '<->' in reaction:
                split = reaction.split(' ')
                split[split.index('<->')] = '->'
                _forward, _reverse = split, split
                forward, reverse = ' '.join(_forward), ' '.join(_reverse[::-1])
                updated_reaction_dict[forward] = {
                    'rconst-names': reaction_dict[reaction]['rconst-values'][0], 
                    'rconst-lb': reaction_dict[reaction]['rconst-lb'][0], 
                    'rconst-ub': reaction_dict[reaction]['rconst-ub'][0],
                    'model': reaction_dict[reaction]['model'][0]
                    }
                updated_reaction_dict[reverse] = {
                    'rconst-names': reaction_dict[reaction]['rconst-values'][1], 
                    'rconst-lb': reaction_dict[reaction]['rconst-lb'][1], 
                    'rconst-ub': reaction_dict[reaction]['rconst-ub'][1],
                    'model': reaction_dict[reaction]['model'][1]
                    }            
            else:
                updated_reaction_dict[reaction] = reaction_dict[reaction]
        return updated_reaction_dict

    def _parse_reaction_dict(self, reaction_dict: dict):
        """
        Static method for processing dictionaries with reactions to
        be modeled with mass action kinetics.
        """

        species = self.species_collection.names

        A, N, rconst_names, rconst_values, rconst_lb, rconst_ub  = [], [], [], [], [], []
        reactions = list(reaction_dict.keys())
        for reaction in reactions:
            rconst_names.append(reaction_dict[reaction]['rconst-names'])
            rconst_values.append(reaction_dict[reaction]['rconst-values'])
            rconst_lb.append(reaction_dict[reaction]['rconst-lb'])
            rconst_ub.append(reaction_dict[reaction]['rconst-ub'])
            b, a = np.zeros(len(species)), np.zeros(len(species))

            #split chemical equation into products and substrates
            _substrates, _products = reaction.split('->')
            _substrates, _products = [re.sub(re.compile(r'\s+'), '', sub).split('*') for sub in _substrates.split('+')], [re.sub(re.compile(r'\s+'), '', prod).split('*') for prod in _products.split('+')]
            for _substrate in _substrates:
                #get substrate stoichiometries and names
                substrate = _substrate[1] if len(_substrate) == 2 else _substrate[0]
                stoichiometry_coeff = int(_substrate[0]) if len(_substrate) == 2 else 1
                a[species.index(substrate)] = stoichiometry_coeff
            for _product in _products:
                #get product stoichiometries and names
                if _product == ['0']:
                    continue
                product = _product[1] if len(_product) == 2 else _product[0]
                stoichiometry_coeff = int(_product[0]) if len(_product) == 2 else 1
                b[species.index(product)] = stoichiometry_coeff
            A.append(a)
            N.append(b-a)
        return MassActionReactions(reactions, A, N, rconst_names, rconst_values, rconst_lb, rconst_ub)

    def initialize(self, concentration_dict: dict, time: Union[list, np.ndarray]):
        """
        Method for defining initial conditions and time values. Also creates the
        update dictionary.
        """

        # parse inputs 
        # assert isinstance(initial_concentrations, dict), 'ChemicalReactionNetwork Error: initial_concentrations must be a dictionary.'
        # assert set([isinstance(k, str) for k in initial_concentrations.keys()]) == set([True]) and set([isinstance(v, numbers.Number) for v in initial_concentrations.values()]) == set([True]), 'ChemicalReactionNetwork Error: keys and values within initial_concentrations must be strings and numerics, respectively.'
        # assert set(initial_concentrations.keys()).issubset(set(self.species)), 'ChemicalReactionNetwork Error: keys within initial_concentrations must be a subset of the species within the reaction network.'
        # assert len(set(initial_concentrations.keys())) == len(list(initial_concentrations.keys())), 'ChemicalReactionNetwork Error: initial_concentrations cannot contain duplicate keys.'
        # assert isinstance(time, (list, np.ndarray)), 'ChemicalReactionNetwork Error: time must be either a list or np.ndarray.'
        # time = np.array(time)
        # assert time.ndim == 1, 'ChemicalReactionNetwork Error: time must be a 1 dimensional array.'
        # assert set([isinstance(v, numbers.Number) for v in time]) == set([True]), 'ChemicalReactionNetwork Error: elements within time must be numerics.'

        self.time, self.timestep = time, time[1] - time[0]
        self.species_collection.set_values_and_bounds(concentration_dict)
        self.update_dictionary = self._create_update_dictionary()

    def _make_update_function(self, index: int, token: str):
        """ 
        Private method for defining functions to update rate constants
        and initial concentrations of chemical reaction network.
        """

        def update(new_value):
            if token == 'rate_constant':
                self.mass_action_reactions.rconst_values[index] = new_value
                self.mass_action_reactions.K[index, index] = new_value
            elif token == 'initial_concentration':
                self.species_collection.values[index] = new_value
        return update

    def _create_update_dictionary(self):
        """ 
        Private method for creating update dictionary. Providing a species or 
        rate constant as a key name will yield a function for updating it.
        """

        mass_action_update, initial_concen_update = {}, {}
        if self.mass_action_reactions.reactions:
            for rate_index, rate_name in enumerate(self.mass_action_reactions.rconst_names):
                mass_action_update[rate_name] = self._make_update_function(rate_index, 'rate_constant')
        for index, specie in enumerate(self.species_collection.names):
            initial_concen_update[specie] = self._make_update_function(index, 'initial_concentration')
        return {**mass_action_update, **initial_concen_update}

    def _define_ODEs(self):
        def ODEs(concentrations: np.ndarray, time: np.ndarray):
            return np.dot(self.mass_action_reactions.N.T, np.dot(self.mass_action_reactions.K, np.prod(np.power(concentrations, self.mass_action_reactions.A), axis=1)))
        return ODEs

    def integrate(
            self, 
            rtol: float = None, 
            atol: float = None, 
            inplace = True, 
            noise_mu: float = 0,
            noise_sigma: float = 0,
            ):
        """ 
        Method for numerical integration of ODE system associated 
        with the reaction network. Outputs nothing, but re-defines 
        the concentrations attribute.

        :param rtol, atol: hyperparameters that control the error tolerance
        of the numerical integrator.
        """

        simulated_data = odeint(self.ODEs, self.species_collection.values, self.time, rtol=rtol, atol=atol).T
        noise_arr = np.random.normal(noise_mu, noise_sigma, simulated_data.shape)
        simulated_data += noise_arr

        if inplace:
            self.simulated_data = simulated_data
        else:
            return simulated_data

class BindingReaction(ChemicalReactionNetwork):
    """
    Child class with methods specific for modelling thermodynamics and kinetics of bimolecular binding reactions.
    """
    def __init__(self, initial_concentrations: dict, reaction_dict: dict, limiting_species: str, ligand: str, equilibtration_time: int, concentration_units='µM', time_units='s', ligand_concentrations=np.insert(np.logspace(-3,2,10), 0, 0)):
        self.equilibration_time = equilibtration_time
        time = np.linspace(0, self.equilibration_time, self.equilibration_time)
        super().__init__(initial_concentrations, reaction_dict, time=time, concentration_units=concentration_units, time_units=time_units)
        self.limiting_species, self.ligand = limiting_species, ligand
        self.limiting_species_index, self.ligand_index = self.species.index(limiting_species), self.species.index(ligand)
        self.complex_index = list({0,1,2} - {self.limiting_species_index, self.ligand_index})[0]

        self.ligand_concentrations = ligand_concentrations


        self.Kd_fit = float
        self.progress_curves = np.ndarray
        self.binding_isotherm = np.ndarray
        self.ground_truth_binding_isotherm = np.ndarray
        self._get_ground_truth_binding_isotherm()

        self._make_equilibration_time_update()

    def _make_equilibration_time_update(self):

        def update(new_value):
            self.equilibration_time = new_value
            self.time = np.linspace(0, self.equilibration_time, self.equilibration_time)

        self.update_dictionary['equilibration_time'] = update
    
    def _get_ground_truth_binding_isotherm(self):
        P = self.initial_concentrations[self.limiting_species_index]
        L = self.ligand_concentrations
        Kd = self.mass_action_reactions.K[1,1] / self.mass_action_reactions.K[0,0]

        term1 = np.full(len(L), P + Kd) + L
        term2 = np.square(term1) - np.vstack([np.full(len(L), 4 * P), L]).prod(axis=0)
        self.ground_truth_binding_isotherm = (term1 - np.sqrt(term2)) / (2 * P)    

    def get_progress_curves_and_isotherm(self, inplace=True):
        progress_curves, binding_isotherm = [], []
        for concen in self.ligand_concentrations:
            self.update_dictionary[self.ligand](concen)

            _progress_curves = self.integrate(self.initial_concentrations, self.time, inplace=False)
            progress_curves.append(_progress_curves[self.complex_index])
            binding_isotherm.append(_progress_curves[self.complex_index, -1] / (_progress_curves[self.complex_index, -1] + _progress_curves[self.limiting_species_index, -1]))

        progress_curves = np.vstack(progress_curves)
        binding_isotherm = np.array(binding_isotherm)

        if inplace:
            self.progress_curves = progress_curves
            self.binding_isotherm = binding_isotherm
        else:
            return progress_curves, binding_isotherm

    def fit_Kd(self):

        x = self.ligand_concentrations
        y = self.binding_isotherm

        P = self.initial_concentrations[self.limiting_species_index]
        def model(L, Kd):
            term1 = np.full(len(L), P + Kd) + L
            term2 = np.square(term1) - np.vstack([np.full(len(L), 4 * P), L]).prod(axis=0)
            return (term1 - np.sqrt(term2)) / (2 * P)    

        _, params = curve_fit(model, x, y)
        self.Kd_fit = params[0]

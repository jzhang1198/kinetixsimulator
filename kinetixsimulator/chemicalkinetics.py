""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for modelling the kinetics of chemical reaction networks.
"""

#imports
import re
import copy
import numbers
import numpy as np
from scipy.integrate import odeint
from typing import Union, List
from scipy.optimize import curve_fit, minimize

class Reaction:
    """ 
    Utility class for organizing data for unidirectional reactions.
    """

    def __init__(
            self,
            reaction_string: str, 
            rconst_name: str,
            rconst_value: float, 
            ):
        
        Reaction._parse_inputs(reaction_string, rconst_name, rconst_value)
        self.reaction_string = reaction_string
        self.rconst_name = rconst_name 
        self.rconst_value = rconst_value

    @staticmethod
    def _parse_inputs(
        reaction_string, 
        rconst_name, 
        rconst_value
        ):

        """ 
        Static method for ensuring input arguments satisfy code assumptions.
        """
        pass

class BidirectionalReaction:
    """ 
    Utility class for organizing data for bidirectional reactions.
    """

    def __init__(
            self, 
            reaction_string: str,
            rconst_names: list,
            rconst_values: list,
        ):
        
        BidirectionalReaction._parse_inputs(reaction_string, rconst_names, rconst_values)
        self.reactions = BidirectionalReaction.split(reaction_string, rconst_names, rconst_values)

    @staticmethod
    def _parse_inputs(
        reaction_string,
        rconst_names,
        rconst_values,
    ):
        pass
    
    @staticmethod
    def split(
        reaction_string, 
        rconst_names, 
        rconst_values, 
        ):
        """ 
        Static method that splits the bidirectional reaction into two unimolecular
        reactions.
        """

        split = reaction_string.split(' ')
        split[split.index('<->')] = '->'
        _forward, _reverse = split, split
        forward, reverse = ' '.join(_forward), ' '.join(_reverse[::-1])

        reactions = [
            Reaction(forward, rconst_names[0], rconst_values[0]),
            Reaction(reverse, rconst_names[1], rconst_values[1])
        ]

        return reactions

class KineticModel:
    """
    Class for an arbitrary network of chemical reactions.

    Attributes
    ----------
    species: list
        A list of all chemical species present.
    reaction_collection: list
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

    def __init__(
            self, 
            time: np.ndarray,
            reactions: List[Union[Reaction, BidirectionalReaction]],
            integrator_atol: float = 1.5e-8,
            integrator_rtol: float = 1.5e-8,
            concentration_units='µM', 
            time_units='s'
            ):
        
        # define an attribute containing only unidirectional reactions, for simplicity
        unidirectional_reactions = [reaction for reaction in reactions if isinstance(reaction, Reaction)]
        bidirectional_reactions = [reaction for reaction in reactions if isinstance(reaction, BidirectionalReaction)]
        for reaction in bidirectional_reactions:
            unidirectional_reactions += reaction.reactions
        self.reactions = unidirectional_reactions

        # define time attributes
        self.time = time 
        self.time_units = time_units

        # define mass action matrices
        A, N, K, species, rconst_names, rconst_values = self.define_mass_action_matrices()
        self.A = A
        self.N = N
        self.K = K
        self.rconst_names = rconst_names
        self.rconst_values = rconst_values 

        # define attributes acessed by the integrator
        self.ODEs = self._define_ODEs()
        self.atol = integrator_atol
        self.rtol = integrator_rtol

        # define species attributes
        self.specie_names = species 
        self.specie_initial_concs = [0] * len(species)
        self.concentration_units = concentration_units

        # define and store update functions in a dict
        self.update_dictionary = self._create_update_dictionary()

        # attributes to be defined
        self.simulated_data = None 

    def define_mass_action_matrices(self):

        characters = {'*', '->', '+', '<->'}
        species = list(set([specie for specie in sum([i.reaction_string.split(' ') for i in self.reactions], []) if specie not in characters and not specie.isnumeric()]))

        A, N, rconst_names, rconst_values  = [], [], [], []
        for reaction in self.reactions:
            rconst_names.append(reaction.rconst_name), 
            rconst_values.append(reaction.rconst_value)

            b, a = np.zeros(len(species)), np.zeros(len(species))

            #split chemical equation into products and substrates
            _substrates, _products = reaction.reaction_string.split('->')
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
        
        A, N, K = np.vstack(A), np.vstack(N), np.diag(np.array(rconst_values))

        return A, N, K, species, rconst_names, rconst_values

    def set_initial_concentration(self, specie_name: str, initial_concentration: float):
        self.update_dictionary[specie_name](initial_concentration)

    def _make_update_function(self, index: int, token: str):
        """ 
        Private method for defining functions to update rate constants
        and initial concentrations of chemical reaction network.
        """

        def update(new_value):
            if token == 'rate_constant':
                self.rconst_values[index] = new_value
                self.K[index, index] = new_value
            elif token == 'initial_concentration':
                self.specie_initial_concs[index] = new_value
        return update

    def _create_update_dictionary(self):
        """ 
        Private method for creating update dictionary. Providing a species or 
        rate constant as a key name will yield a function for updating it.
        """

        rconst_update, specie_initial_conc_update = {}, {}
        if len(self.reactions) > 0:
            for rate_index, rate_name in enumerate(self.rconst_names):
                rconst_update[rate_name] = self._make_update_function(rate_index, 'rate_constant')
        for index, specie in enumerate(self.specie_names):
            specie_initial_conc_update[specie] = self._make_update_function(index, 'initial_concentration')
        return {**rconst_update, **specie_initial_conc_update}

    def _define_ODEs(self):
        def ODEs(concentrations: np.ndarray, time: np.ndarray):
            return np.dot(self.N.T, np.dot(self.K, np.prod(np.power(concentrations, self.A), axis=1)))
        return ODEs

    def simulate(
            self, 
            inplace = True, 
            noise_mu: float = 0,
            noise_sigma: float = 0,
            observable_species: list = []
            ):
        """ 
        Method for numerical integration of ODE system associated 
        with the reaction network. Outputs nothing, but re-defines 
        the concentrations attribute.

        :param rtol, atol: hyperparameters that control the error tolerance
        of the numerical integrator.
        """

        simulated_data = odeint(self.ODEs, self.specie_initial_concs, self.time, rtol=self.rtol, atol=self.atol).T
        noise_arr = np.random.normal(noise_mu, noise_sigma, simulated_data.shape)
        simulated_data += noise_arr

        if len(observable_species) > 0:
            observable_indices = [self.specie_names.index(specie) for specie in observable_species]
            simulated_data = simulated_data[observable_indices]

        if inplace:
            self.simulated_data = simulated_data
        else:
            return simulated_data

class KineticFitter:
    """ 
    Class for global fitting of simulated or collected kinetic data.

    Attributes
    ----------
    kinetic_model: KineticModel
        The model to which the data is fit to. Can also be used to
        generate simulated data under various conditions.
    """

    def __init__(self, kinetic_model: KineticModel):
        self.kinetic_model = copy.deepcopy(kinetic_model) 
        self.dataset = []
        self.objective = None

        # obtain initial guesses and constraints from kinetic model
        self.rconst_names = self.kinetic_model.reaction_collection.rconst_names
        self.rconst_p0 = self.kinetic_model.reaction_collection.rconst_values
        self.rconst_bounds = [(lb, ub) for lb, ub in zip(self.kinetic_model.reaction_collection.rconst_lb, self.kinetic_model.reaction_collection.rconst_ub)]
        self.rconst_fits = None
        self.fit_result = None

    def add_data(self, conditions: dict, data: np.ndarray, observable_species: list = []):
        """ 
        Method for adding simulated or experimental data.
        """

        observable_indices = [self.kinetic_model.specie_collection.names.index(specie) for specie in observable_species]
        self.dataset.append((conditions, data, observable_indices))  

    def fit_model(self):
        """ 
        Method for fitting rate constants to the observed data.
        """

        # dynamically define the objective
        def objective(rconsts):

            residuals = 0

            # update kinetic model with sampled rate constants
            for index, rconst in enumerate(rconsts):
                self.kinetic_model.update_dictionary[self.rconst_names[index]] = rconst

            # iterate through items in the dataset, simulate, and compute residuals
            for item in self.dataset:
                conditions, data, observable_indices = item

                # update kinetic model with initial concentrations
                for specie in self.kinetic_model.specie_collection.names:
                    self.kinetic_model.update_dictionary[specie] = 0 if specie not in conditions.keys() else conditions[specie]

                simulated_data = self.kinetic_model.simulate(inplace=False)[observable_indices]
                residuals += np.square(np.subtract(simulated_data, data)).sum()

            return residuals
        
        self.objective = objective
        
        fit_result = minimize(objective, self.rconst_p0, bounds=self.rconst_bounds)
        self.fit_result = fit_result
        self.rconst_fits = fit_result.x

    def generate_1D_SSE_surface(self):

        SSE_surface_1D = {}
        for rconst, rconst_fit in zip(self.rconst_names, self.rconst_fits):
            rconst_space = np.linspace(rconst_fit - 0.2 * rconst_fit, 0.2 * rconst_fit + rconst_fit, 40)
            SSE = [self.objective(r) for r in rconst_space]
            


        return SSE_surface_1D

    def generate_2D_SSE_surface(self):
        pass

class BindingReaction(KineticModel):
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
        Kd = self.reaction_collection.K[1,1] / self.reaction_collection.K[0,0]

        term1 = np.full(len(L), P + Kd) + L
        term2 = np.square(term1) - np.vstack([np.full(len(L), 4 * P), L]).prod(axis=0)
        self.ground_truth_binding_isotherm = (term1 - np.sqrt(term2)) / (2 * P)    

    def get_progress_curves_and_isotherm(self, inplace=True):
        progress_curves, binding_isotherm = [], []
        for concen in self.ligand_concentrations:
            self.update_dictionary[self.ligand](concen)

            _progress_curves = self.simulate(self.initial_concentrations, self.time, inplace=False)
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

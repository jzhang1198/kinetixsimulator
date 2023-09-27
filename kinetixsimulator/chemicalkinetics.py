""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for modelling the kinetics of chemical reaction networks.
"""

#imports
import re
import copy
import numbers
import numpy as np
from typing import Union, List
from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize

class Reaction:
    """ 
    Container class for organizing data for unidirectional reactions.
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

        assert isinstance(reaction_string, str), 'Reaction Error: reaction_string must be a string.'
        assert '->' in reaction_string and '<->' not in reaction_string, 'Reaction Error: Substrates and products within reaction_string must be separated by a "->" character.'
        assert isinstance(rconst_name, str), 'Reaction Error: rconst_name must be a string.'
        assert isinstance(rconst_value, numbers.Number) and rconst_value > 0, 'Reaction Error: rconst_value must be a positive numeric.' 

class ReversibleReaction:
    """ 
    Container class for organizing data for reversible reactions.
    """

    def __init__(
            self, 
            reaction_string: str,
            rconst_names: list,
            rconst_values: list,
        ):
        
        ReversibleReaction._parse_inputs(reaction_string, rconst_names, rconst_values)
        self.reaction_string = reaction_string
        self.rconst_names = rconst_names
        self.rconst_values = rconst_values

    @staticmethod
    def _parse_inputs(
        reaction_string,
        rconst_names,
        rconst_values,
    ):
        
        assert isinstance(reaction_string, str), 'ReversibleReaction Error: reaction_string must be a string.'
        assert '<->' in reaction_string and '->' not in reaction_string, 'ReversibleReaction Error: Substrates and products within reaction_string must be separated by a "<->" character.'
        assert isinstance(rconst_names, list) and set([True]) == set([isinstance(name, str) for name in rconst_names]), 'ReversibleReaction Error: rconst_names must be a list of strings.'
        assert isinstance(rconst_values, list) and set([True]) == set([isinstance(value, numbers.Number) for value in rconst_values]), 'ReversibleReaction Error: rconst_values must be a list of numerics.'
        assert len(rconst_values) == len(rconst_names), 'ReversibleReaction Error: rconst_names and rconst_values must be the same length.'
    
    def split(self):
        """ 
        Method that splits the reversible reaction into two unimolecular
        reactions.
        """

        split = self.reaction_string.split(' ')
        split[split.index('<->')] = '->'
        _forward, _reverse = split, split
        forward, reverse = ' '.join(_forward), ' '.join(_reverse[::-1])

        reactions = [
            Reaction(forward, self.rconst_names[0], self.rconst_values[0]),
            Reaction(reverse, self.rconst_names[1], self.rconst_values[1])
        ]

        return reactions

class MMReaction:
    """ 
    Container class for storing data for reactions modeled with 
    Michaelis-Menten kinetics.
    """

    def __init__(
            self, 
            reaction_string: str, 
            Km_name: str, 
            Km_value: float, 
            kcat_name: str, 
            kcat_value: float
        ):
                    
        self.reaction_string = reaction_string 
        self.Km_name = Km_name
        self.Km_value = Km_value
        self.kcat_name = kcat_name
        self.kcat_value = kcat_value

    @staticmethod
    def _parse_inputs(
        reaction_string,
        Km_name,
        Km_value,
        kcat_name,
        kcat_value
    ):
        
        assert isinstance(reaction_string, str), 'MMReaction Error: reaction_string must be a string.'
        assert '<->' in reaction_string and '->' in reaction_string, 'MMReaction Error: Substrates and products for the binding and catalysis steps must be separated with "<->" and "->" characters, respectively.'
        assert isinstance(Km_name, str), 'MMReaction Error: Km_name must be a string.'
        assert isinstance(Km_value, numbers.Number) and Km_value > 0, 'MMReaction Error: Km_value must be a positive numeric.' 
        assert isinstance(kcat_name, str), 'MMReaction Error: kcat_name must be a string.'
        assert isinstance(kcat_value, numbers.Number) and kcat_value > 0, 'MMReaction Error: kcat_value must be a positive numeric.' 

    def split(self):

        # update reaction strings
        binding_reaction_string = self.reaction_string.split(' ->')[0]
        chemical_reaction_string = self.reaction_string.split('<-> ')[-1]

        split = binding_reaction_string.split(' ')
        split[split.index('<->')] = '->'
        _forward, _reverse = split, split
        forward, reverse = ' '.join(_forward), ' '.join(_reverse[::-1])

        return [forward, reverse, chemical_reaction_string]
        
class KineticModel:
    """
    Class for an arbitrary network of chemical reactions.
    """

    def __init__(
            self, 
            time: np.ndarray,
            reactions: List[Union[Reaction, ReversibleReaction]],
            integrator_atol: float = 1.5e-8,
            integrator_rtol: float = 1.5e-8,
            concentration_units='uM', 
            time_units='s'
            ):
        
        # segregate reactions
        unimolecular_reactions = [reaction for reaction in reactions if isinstance(reaction, Reaction)]
        unimolecular_reactions += np.array([reaction.split() for reaction in reactions if isinstance(reaction, ReversibleReaction)]).flatten().tolist()
        MM_reactions = [reaction for reaction in reactions if isinstance(reaction, MMReaction)]
        self.reactions = unimolecular_reactions
        self.MM_reactions = MM_reactions
        
        # define time attributes
        self.time = time 
        self.time_units = time_units

        # define specie attributes
        self.species = self._get_species()
        self.specie_initial_concs = [0] * len(self.species)
        self.concentration_units = concentration_units

        # define diffusion-limited rate consistent with provided units
        self.kon = self._set_kon()

        # define reaction attributes
        self.A, self.N, self.rconst_names, self.rconst_values, self.MM_rconst_names, self.MM_rconst_values = self._get_rconsts_and_stoichiometries()
        self.K = None

        # define attributes acessed by the integrator
        self.atol = integrator_atol
        self.rtol = integrator_rtol
        self.ODEs = self._define_ODEs()

        # define and store update functions in a dict
        self.update_dictionary = self._create_update_dictionary()

        # dynamic attribute to hold simulated data
        self.simulated_data = None 

    def _set_kon(self):
        """ 
        Method to set 
        """

        def_kon = 1e2

        conc_unit_mapper = {
            "M": 1e6,
            "mM": 1e3,
            "uM": 1,
            "nM": 1e-3,
            "pM": 1e-6,
            "fM": 1e-9
        }

        time_unit_mapper = {
            "day": 86400,
            'hour': 3600,
            'minute': 60,
            's': 1,
            'ms': 1e-3,
            'us': 1e-3,
            'ns': 1e-3,
            'ps': 1e-3,
            'fs': 1e-3,
        }

        uM_conversion = conc_unit_mapper[self.concentration_units]
        s_conversion = time_unit_mapper[self.time_units]

        kon = def_kon * uM_conversion * s_conversion
        return kon

    def _get_species(self):
        characters = {'*', '->', '+', '<->'}
        species = list(set([specie for specie in sum([i.reaction_string.split(' ') for i in self.reactions + self.MM_reactions], []) if specie not in characters and not specie.isnumeric()]))
        return species

    @staticmethod
    def _parse_reaction_string(reaction_string: str, a: np.ndarray, b: np.ndarray, species: list):

        #split chemical equation into products and substrates
        _substrates, _products = reaction_string.split('->')
        _substrates, _products = [re.sub(re.compile(r'\s+'), '', sub).split('*') for sub in _substrates.split('+')], [re.sub(re.compile(r'\s+'), '', prod).split('*') for prod in _products.split('+')]
        for _substrate in _substrates:
            # get substrate stoichiometries and names
            substrate = _substrate[1] if len(_substrate) == 2 else _substrate[0]
            stoichiometry_coeff = int(_substrate[0]) if len(_substrate) == 2 else 1
            a[species.index(substrate)] = stoichiometry_coeff

        for _product in _products:
            # get product stoichiometries and names
            if _product == ['0']:
                continue
            product = _product[1] if len(_product) == 2 else _product[0]
            stoichiometry_coeff = int(_product[0]) if len(_product) == 2 else 1
            b[species.index(product)] = stoichiometry_coeff

        return a, b

    def _get_rconsts_and_stoichiometries(self):

        A, N, rconst_names, rconst_values  = [], [], [], []
        for reaction in self.reactions:
            rconst_names.append(reaction.rconst_name), 
            rconst_values.append(reaction.rconst_value)
            a, b = np.zeros(len(self.species)), np.zeros(len(self.species))
            a, b = KineticModel._parse_reaction_string(reaction.reaction_string, a, b, self.species)
            A.append(a)
            N.append(b-a)
        
        MM_rconst_names, MM_rconst_values = [], []
        for reaction in self.MM_reactions:
            MM_rconst_names.append(reaction.Km_name), MM_rconst_names.append(reaction.kcat_name)
            MM_rconst_values.append(reaction.Km_value), MM_rconst_values.append(reaction.kcat_value)

            for index, elementary_reaction_string in enumerate(reaction.split()):
                a, b = np.zeros(len(self.species)), np.zeros(len(self.species))
                a, b = KineticModel._parse_reaction_string(elementary_reaction_string, a, b, self.species)

                # check to make sure MM reaction is single substrate
                if index == 0:
                    assert a.sum() == 2 and b.sum() == 1, 'MMReaction Error: Reaction must follow a single substrate Michaelis-Menten mechanism.'

                A.append(a)
                N.append(b-a)

        A, N = np.vstack(A), np.vstack(N)
        return A, N, rconst_names, rconst_values, MM_rconst_names, MM_rconst_values

    def set_initial_concentration(self, specie_name: str, initial_concentration: float):
        """ 
        Method for setting the initial concentration of a chemical specie.
        """

        assert isinstance(specie_name, str) and isinstance(initial_concentration, numbers.Number), 'KineticModel Error: specie_name and initial_concentration arguments must be a string or numeric, respectively.'
        assert specie_name in self.species, f'KineticModel Error: cannot set the initial concentration of {specie_name}.'
        self.update_dictionary[specie_name](initial_concentration)

    def _make_update_function(self, index: int, token: str):
        """ 
        Private method for defining functions to update rate constants
        and initial concentrations of chemical reaction network.
        """

        def update(new_value):
            if token == 'rate_constant':
                self.rconst_values[index] = new_value
            elif token =='MM_rate_constant':
                self.MM_rconst_values[index] = new_value
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

        if len(self.MM_reactions) > 0:
            for rate_index, rate_name in enumerate(self.MM_rconst_names):
                rconst_update[rate_name] = self._make_update_function(rate_index, 'MM_rate_constant')

        for index, specie in enumerate(self.species):
            specie_initial_conc_update[specie] = self._make_update_function(index, 'initial_concentration')
        return {**rconst_update, **specie_initial_conc_update}

    def _define_ODEs(self):
        def ODEs(concentrations: np.ndarray, time: np.ndarray):
            return np.dot(self.N.T, np.dot(self.K, np.prod(np.power(concentrations, self.A), axis=1)))
        return ODEs

    def _set_K(self):

        rconsts = copy.deepcopy(self.rconst_values)

        # find microscopic rate constants consistent with MM model
        if len(self.MM_rconst_names) > 0:
            Km_values = np.array(self.MM_rconst_values[0 :: 2])
            kcat_values = np.array(self.MM_rconst_values[1 :: 2])
            kon_values = np.array([self.kon] * len(Km_values))
            koff_values = np.subtract(np.multiply(Km_values, kon_values), kcat_values)
            MM_rconsts = np.zeros(len(self.MM_reactions) * 3)
            MM_rconsts[0::3], MM_rconsts[1::3], MM_rconsts[2::3] = kon_values, koff_values, kcat_values
            rconsts += MM_rconsts.tolist()

        return np.diag(rconsts)
            
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
        the simulated_data attribute.
        """

        self.K = self._set_K()
        simulated_data = odeint(self.ODEs, self.specie_initial_concs, self.time, rtol=self.rtol, atol=self.atol).T
        noise_arr = np.random.normal(noise_mu, noise_sigma, simulated_data.shape)
        simulated_data += noise_arr

        if len(observable_species) > 0:
            observable_indices = [self.species.index(specie) for specie in observable_species]
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
        self.rconst_names = self.kinetic_model.rconst_names + self.kinetic_model.MM_rconst_names
        self.rconst_p0 = self.kinetic_model.rconst_values + self.kinetic_model.MM_rconst_values
        self.rconst_lb = np.zeros(len(self.rconst_p0))
        self.rconst_ub = np.full(len(self.rconst_p0), np.inf)
        self.rconst_fits = None
        self.fit_result = None

    def add_data(self, conditions: dict, data: np.ndarray, observable_species: list = []):
        """ 
        Method for adding simulated or experimental data.
        """

        observable_indices = [self.kinetic_model.species.index(specie) for specie in observable_species]
        self.dataset.append((conditions, data, observable_indices))  

    def set_bounds(self, rconst_name: str, lb: float, ub: float):
        index = self.rconst_names.index(rconst_name)
        self.rconst_lb[index] = lb
        self.rconst_ub[index] = ub

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
                for specie in self.kinetic_model.species:
                    self.kinetic_model.update_dictionary[specie] = 0 if specie not in conditions.keys() else conditions[specie]

                simulated_data = self.kinetic_model.simulate(inplace=False)[observable_indices]
                residuals += np.square(np.subtract(simulated_data, data)).sum()

            return residuals
        
        self.objective = objective
        
        fit_result = minimize(objective, self.rconst_p0, bounds=[(lb, ub) for lb, ub in zip(self.rconst_lb, self.rconst_ub)])
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
    Deprecation warning: DeprecatedClass is no longer maintained and will be removed.
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
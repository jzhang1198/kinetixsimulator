""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for modelling the kinetics of chemical reaction networks.
"""

#imports
import re
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

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

    def __init__(self, initial_concentrations: dict, reaction_dict: dict, time=np.linspace(0,100,100), concentration_units='µM', time_units='s'):
        characters = {'*', '->', '+', '<->'}

        if len(reaction_dict) < 1:
            print('no reaction')

        self.species = list(set([specie for specie in sum([i.split(' ') for i in reaction_dict.keys()], []) if specie not in characters and not specie.isnumeric()])) #parses through chemical equation strings to find all unique species
        self._reaction_dict = ChemicalReactionNetwork._process_reaction_dict(reaction_dict)
        self.mass_action_reactions = ChemicalReactionNetwork._parse_reaction_dict(self._reaction_dict, self.species)
        self.time, self.timestep = time, time[1] - time[0]
        self.initial_concentrations = np.array([initial_concentrations[specie] if specie in initial_concentrations.keys() else 1e-50 for specie in self.species]) #generates initial values based on initial_concentrations
        self.update_dictionary = self._create_update_dictionary()
        self.ODEs = self._define_ODEs()
        self.concentrations = np.ndarray #create a placeholder for ODEs and concentrations attribute
        self.concentration_units, self.time_units = concentration_units, time_units

    @staticmethod
    def _process_reaction_dict(reaction_dict: dict):
        """
        Splits reversible reactions into elementary reactions.
        """
        updated_reaction_dict = {}
        for reaction in reaction_dict.keys():

            if 'fit' not in reaction_dict[reaction].keys():
                reaction_dict[reaction]['fit'] = False

            difference = {'rate-constants', 'rate-constant-names'}.difference(set(reaction_dict[reaction].keys()))
            if len(difference) != 0:
                message = '{reaction} missing the following keys: {keys}'
                raise utils.MalformedRxnError(message.format(reaction=reaction, keys=', '.join(list(difference))))

            if '<->' in reaction:
                forward_constant, reverse_constant = reaction_dict[reaction]['rate-constant-names']
                split = reaction.split(' ')
                split[split.index('<->')] = '->'
                _forward, _reverse = split, split
                forward, reverse = ' '.join(_forward), ' '.join(_reverse[::-1])
                updated_reaction_dict[forward] = {'rate-constants': reaction_dict[reaction]['rate-constants'][0], 'rate-constant-names': reaction_dict[reaction]['rate-constant-names'][0]}
                updated_reaction_dict[reverse] = {'rate-constants': reaction_dict[reaction]['rate-constants'][1], 'rate-constant-names': reaction_dict[reaction]['rate-constant-names'][1]}
            else:
                updated_reaction_dict[reaction] = reaction_dict[reaction]
        return updated_reaction_dict

    @staticmethod
    def _parse_reaction_dict(reaction_dict: dict, species: list):
        """
        Static method for processing dictionaries with reactions to
        be modeled with mass action kinetics.
        """

        #if no mass action reactions, instantiate a dummy
        if len(reaction_dict) == 0:
            return MassActionReactions()

        A, N, rate_names, rates = [], [], [], []
        reactions = list(reaction_dict.keys())
        for reaction in reactions:
            rate_names.append(reaction_dict[reaction]['rate-constant-names']), rates.append(reaction_dict[reaction]['rate-constants'])
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
        return MassActionReactions(reactions, A, N, rate_names, rates)

    def _make_update_function(self, index: int, token: str):
        """ 
        Private method for defining functions to update rate constants
        and initial concentrations of chemical reaction network.
        """

        def update(new_value):
            if token == 'rate_constant':
                self.mass_action_reactions.K[index, index] = new_value
            elif token == 'initial_concentration':
                self.initial_concentrations[index] = new_value
        return update

    def _create_update_dictionary(self):
        """ 
        Private method for creating update dictionary. Providing a species or 
        rate constant as a key name will yield a function for updating it.
        """

        mass_action_update, initial_concen_update = {}, {}
        if self.mass_action_reactions.reactions:
            for rate_index, rate_name in enumerate(self.mass_action_reactions.rate_names):
                mass_action_update[rate_name] = self._make_update_function(rate_index, 'rate_constant')
        for index, specie in enumerate(self.species):
            initial_concen_update[specie] = self._make_update_function(index, 'initial_concentration')
        return {**mass_action_update, **initial_concen_update}

    def _define_ODEs(self):
        def ODEs(concentrations: np.ndarray, time: np.ndarray):
            return np.dot(self.mass_action_reactions.N.T, np.dot(self.mass_action_reactions.K, np.prod(np.power(concentrations, self.mass_action_reactions.A), axis=1)))
        return ODEs

    def integrate(self, initial_concentrations: np.ndarray, time: np.ndarray, rtol=None, atol=None, inplace=True):
        """ 
        Method for numerical integration of ODE system associated 
        with the reaction network. Outputs nothing, but re-defines 
        the concentrations attribute.

        :param rtol, atol: hyperparameters that control the error tolerance
        of the numerical integrator.
        """
        if inplace:
            self.concentrations = odeint(self.ODEs, initial_concentrations, time, rtol=rtol, atol=atol).T
        else:
            return odeint(self.ODEs, initial_concentrations, time, rtol=rtol, atol=atol).T

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

class MassActionReactions:
    """ 
    Class for reactions modelled with mass action kinetics.

    Attributes
    ----------
    reactions: list
        A list of chemical equations represented as strings.
    A: np.ndarray
        The substrate stoichiometry matrix for the mass action reactions.
    N: np.ndarray
        The reaction stoichiometry matrix for the mass action reactions.
    rate_names: list
        A list of rate names for each rate constant.
    K: np.ndarray
        The rate matrix for the mass action reactions.
    fit: np.ndarray
        Array designating whether a reaction is to be fit.
    """

    def __init__(self, reactions=None, A=None, N=None, rate_names=None, rates=None):
        args =  [reactions, A, N, rate_names, rates]
        types = [type(arg) for arg in args]

        #if no arguments passed, instantiate a dummy
        if set(types) == set([type(None)]):
            self.reactions, self.N, self.A, self.rate_names, self.K = [None] * 5

        #otherwise, instantiate a bonafide object
        elif set(types) == set([list]):
            self.reactions = reactions
            self.N = np.vstack(N)
            self.A = np.vstack(A)
            self.rate_names = rate_names
            self.K = np.diag(np.array(rates))

        else:
            #set up handling of exceptions
            pass

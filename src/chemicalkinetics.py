""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for modelling the kinetics of chemical reaction networks.
"""

#imports
import re
from . import utils
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

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
        self.mass_action_reactions, self.michaelis_menten_reactions = ChemicalReactionNetwork._parse_reaction_dict(self._reaction_dict, self.species)
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

            difference = {'model', 'rate-constants', 'rate-constant-names'}.difference(set(reaction_dict[reaction].keys()))
            if len(difference) != 0:
                message = '{reaction} missing the following keys: {keys}'
                raise utils.MalformedRxnError(message.format(reaction=reaction, keys=', '.join(list(difference))))

            if '<->' in reaction:
                forward_constant, reverse_constant = reaction_dict[reaction]['rate-constant-names']
                split = reaction.split(' ')
                split[split.index('<->')] = '->'
                _forward, _reverse = split, split
                forward, reverse = ' '.join(_forward), ' '.join(_reverse[::-1])
                updated_reaction_dict[forward] = {'rate-constants': reaction_dict[reaction]['rate-constants'][0], 'model': reaction_dict[reaction]['model'], 'rate-constant-names': reaction_dict[reaction]['rate-constant-names'][0]}
                updated_reaction_dict[reverse] = {'rate-constants': reaction_dict[reaction]['rate-constants'][1], 'model': reaction_dict[reaction]['model'], 'rate-constant-names': reaction_dict[reaction]['rate-constant-names'][1]}
            else:
                updated_reaction_dict[reaction] = reaction_dict[reaction]
        return updated_reaction_dict

    @staticmethod
    def _parse_reaction_dict(reaction_dict: dict, species: list):
        mass_action_dict = dict([(key, value) for key, value in zip(reaction_dict.keys(), reaction_dict.values()) if reaction_dict[key]['model'] == 'mass-action'])
        michaelis_menten_dict = dict([(key, value) for key, value in zip(reaction_dict.keys(), reaction_dict.values()) if reaction_dict[key]['model'] == 'michaelis-menten'])
        return ChemicalReactionNetwork._parse_mass_action(mass_action_dict, species), ChemicalReactionNetwork._parse_michaelis_menten(michaelis_menten_dict, species)

    @staticmethod
    def _parse_mass_action(mass_action_dict: dict, species: list):
        """
        Static method for processing dictionaries with reactions to
        be modeled with mass action kinetics.
        """

        #if no mass action reactions, instantiate a dummy
        if len(mass_action_dict) == 0:
            return MassActionReactions()

        A, N, rate_names, rates = [], [], [], []
        reactions = list(mass_action_dict.keys())
        for reaction in reactions:
            rate_names.append(mass_action_dict[reaction]['rate-constant-names']), rates.append(mass_action_dict[reaction]['rate-constants'])
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

    @staticmethod
    def _parse_michaelis_menten(michaelis_menten_dict: dict, species: list):
        """
        Static method for processing dictionaries with reactions to
        be modeled with MM kinetics.
        """

        #if no MM reactions, instantiate a dummy
        if len(michaelis_menten_dict) == 0:
            return MichaelisMentenReactions()
        
        substrates, enzymes, products, Kms, kcats = [], [], [], [], []
        reactions, values = michaelis_menten_dict.keys(), michaelis_menten_dict.values()
        for reaction, value in zip(reactions, values):
            Km_key, kcat_key = value['rate-constant-names']

            #split reaction equation into substrates and products
            _left, _right = reaction.split('->')
            left, right = [re.sub(re.compile(r'\s+'), '', sub).split('*') for sub in _left.split('+')], [re.sub(re.compile(r'\s+'), '', prod).split('*') for prod in _right.split('+')]
            left_species, left_stoichios, right_species, right_stoichios = [], [], [], []
            for specie in left:
                name = specie[1] if len(specie) == 2 else specie[0]
                left_species.append(name)
                stoichiometry_coeff = int(specie[0]) if len(specie) == 2 else 1
                left_stoichios.append(stoichiometry_coeff)
            for specie in right:
                name = specie[1] if len(specie) == 2 else specie[0]
                right_species.append(name)
                stoichiometry_coeff = int(specie[0]) if len(specie) == 2 else 1
                right_stoichios.append(stoichiometry_coeff)
                    
            enzyme = list(set(left_species).intersection(set(right_species)))[0] #enzyme is present on both sides of equation
            substrate = list(set(left_species) - set([enzyme]))[0] 
            product = list(set(right_species) - set([enzyme]))[0] 
            substrate_index, enzyme_index, product_index = species.index(substrate), species.index(enzyme), species.index(product)
            substrate_stoichiometry, product_stoichiometry = left_stoichios[left_species.index(substrate)], right_stoichios[right_species.index(product)]

            substrates.append((substrate_index, substrate_stoichiometry))
            products.append((product_index, product_stoichiometry))
            enzymes.append(enzyme_index)
            Kms.append((Km_key, value['rate-constants'][0]))
            kcats.append((kcat_key, value['rate-constants'][1]))
        return MichaelisMentenReactions(list(reactions), substrates, enzymes, products, Kms, kcats)

    def _make_update_function(self, index: int, token: str):
        """ 
        Private method for defining functions to update rate constants
        and initial concentrations of chemical reaction network.
        """

        def update(new_value):
            if token == 'rate_constant':
                self.mass_action_reactions.K[index, index] = new_value
            elif token == 'Km':
                self.michaelis_menten_reactions.Kms[index] = new_value
            elif token == 'kcat':
                self.michaelis_menten_reactions.kcats[index] = new_value
            elif token == 'initial_concentration':
                self.initial_concentrations[index] = new_value
        return update

    def _create_update_dictionary(self):
        """ 
        Private method for creating update dictionary. Providing a species or 
        rate constant as a key name will yield a function for updating it.
        """

        mass_action_update, michaelis_menten_update, initial_concen_update = {}, {}, {}
        if self.mass_action_reactions.reactions:
            for rate_index, rate_name in enumerate(self.mass_action_reactions.rate_names):
                mass_action_update[rate_name] = self._make_update_function(rate_index, 'rate_constant')
        if self.michaelis_menten_reactions.reactions:
            for index, (Km_name, kcat_name) in enumerate(zip(self.michaelis_menten_reactions.Km_names, self.michaelis_menten_reactions.kcat_names)):
                michaelis_menten_update[Km_name], michaelis_menten_update[kcat_name] = self._make_update_function(index, 'Km'), self._make_update_function(index, 'kcat')
        for index, specie in enumerate(self.species):
            initial_concen_update[specie] = self._make_update_function(index, 'initial_concentration')
        return {**mass_action_update, **michaelis_menten_update, **initial_concen_update}

    def _define_ODEs(self):
        if type(self.mass_action_reactions.reactions) == list:
            def compute_mass_action_rates(concentrations):
                return np.dot(self.mass_action_reactions.N.T, np.dot(self.mass_action_reactions.K, np.prod(np.power(concentrations, self.mass_action_reactions.A), axis=1)))
        elif self.mass_action_reactions.reactions == None:
            empty = np.zeros(len(self.species))
            def compute_mass_action_rates(concentrations):
                return empty
        if type(self.michaelis_menten_reactions.reactions) == list:
            def compute_michaelis_menten_rates(concentrations, michaelis_menten_reactions):
                return michaelis_menten_reactions.compute_velocities(concentrations)
        elif self.michaelis_menten_reactions.reactions == None:
            empty = np.zeros(len(self.species))
            def compute_michaelis_menten_rates(concentrations, michaelis_menten_reactions):
                return empty

        def ODEs(concentrations: np.ndarray, time: np.ndarray, michaelis_menten_reactions):
            return np.vstack((compute_michaelis_menten_rates(concentrations, michaelis_menten_reactions), compute_mass_action_rates(concentrations))).sum(axis=0)
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
            self.concentrations = odeint(self.ODEs, initial_concentrations, time, args=(self.michaelis_menten_reactions,), rtol=rtol, atol=atol).T
        else:
            return odeint(self.ODEs, initial_concentrations, time, args=(self.michaelis_menten_reactions,), rtol=rtol, atol=atol).T

    def _get_fitting_params(self, params: list):
        indices = np.array([])
        for param in params:
            mass_action_rates = np.array(self.mass_action_reactions.rate_names)
            Km_names = np.array(self.michaelis_menten_reactions.Km_names)
            kcat_names = np.array(self.michaelis_menten_reactions.kcat_names)

            mass_action_indices = np.argwhere(mass_action_rates == param).flatten()
            Km_indices = np.argwhere(Km_names == param).flatten()
            kcat_indices = np.argwhere(kcat_names == param).flatten()
            indices = np.hstack([indices, mass_action_indices, Km_indices, kcat_indices])
            if len(Km_indices) > 0 or len(kcat_indices) > 0:
                return 1
        return indices

    def _generate_ground_truth_data(self, fitting_concentrations: np.array, variable: str, observable: str):
        ground_truth_data = []
        for fitting_concen in fitting_concentrations:
            self.update_dictionary[variable](fitting_concen)
            ground_truth_data.append(self.integrate(self.initial_concentrations, self.time, inplace=False)[self.species.index(observable)])
        return np.vstack(ground_truth_data)

    def fit(self, variable: str, observable: str, params: list, fitting_concentrations: np.array):
        param_indices = self._get_fitting_params(params)
        if type(param_indices) == int:
            print('Fitting not supported for Michaelis Menten parameters in this version. This functionality will be included in a later version!')
            return

        ground_truth_data = self._generate_ground_truth_data(fitting_concentrations, variable, observable)

        def objective(K, object):
            observable_index = object.species.index(observable)
            observable_concentrations = []
            
            # update attributes for parameter attributes
            for param_value, param_name in zip(K, params):
                object.update_dictionary[param_name](param_value)

            # update initial concentrations of variable species and integrate
            for concentration in fitting_concentrations:
                object.update_dictionary[variable](concentration)
                observable_concentrations.append(object.integrate(object.initial_concentrations, object.time, inplace=False)[observable_index])

            # compute sum square error
            observable_concentrations = np.vstack(observable_concentrations)
            sse = np.square(observable_concentrations - ground_truth_data).mean()
            return sse

        K0 = np.ones(len(params))
        bounds = [(0, None) for i in range(len(params))]
        result = minimize(objective, K0, args=(self), bounds=bounds)
        return result

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
            # self.NK = np.dot(self.N.T, self.K)

        else:
            #set up handling of exceptions
            pass

class MichaelisMentenReactions:
    """ 
    Class for reactions modelled with MM kinetics.

    Attributes
    ----------
    reactions: list
        A list of chemical equations represented as strings.
    substrate_indices: np.ndarray 
        Indices of substrate species. Indexed by reaction.
    substrate_stoichiometries: np.ndarray
        Stoichiometries of substrates. Indexed by reaction.
    product_indices: np.ndarray 
        Indices of product species. Indexed by reaction.
    product_stoichiometries: np.ndarray 
        Stoichiometries of products. Indexed by reaction.
    enzyme_indices: np.ndarray
        Indices of enzyme species. Indexed by reaction.
    Km_names: np.ndarray
        Names of Km constants for each reaction.
    Kms: np.ndarray
        Values of Km constants for each reaction.
    kcat_names: np.ndarray
        Names of kcat constants for each reaction.
    kcats: np.ndarray
        Values of kcat constants for each reaction.
    """

    def __init__(self, reactions=None, substrates=None, enzymes=None, products=None, Kms=None, kcats=None):
        args = [reactions, substrates, enzymes, products, Kms, kcats]
        types = [type(arg) for arg in args]

        #if no arguments passed, instantiate a dummy
        if set(types) == set([type(None)]):
            self.reactions, self.substrate_indices, self.substrate_stoichiometries, self.product_indices, self.product_stoichiometries, \
            self.enzyme_indices, self.Km_names, self.Kms, self.kcat_names, self.kcats = [None] * 10

        #otherwise, instantiate a bonafide object
        elif set(types) == set([list]):
            self.reactions = reactions
            self.substrate_indices, self.substrate_stoichiometries = map(np.array,zip(*substrates))
            self.product_indices, self.product_stoichiometries = map(np.array,zip(*products))
            self.enzyme_indices = enzymes
            self.Km_names, self.Kms = map(np.array,zip(*Kms))
            self.kcat_names, self.kcats = map(np.array,zip(*kcats))

        else:
            #include something for error handling   
            pass
        
    def compute_velocities(self, concentrations: np.ndarray):
        """ 
        Function for vectorized calculation of MM rates using the
        quadratic MM equation (no free ligand approximation).
        """

        substrate_velocities, product_velocities = np.zeros(len(concentrations)), np.zeros(len(concentrations))
        substrate_vector = concentrations[self.substrate_indices]   
        enzyme_vector = concentrations[self.enzyme_indices] 

        term1 = np.vstack((substrate_vector, enzyme_vector, self.Kms)).sum(axis=0)
        t1, t2 = np.square(term1), np.vstack((enzyme_vector, substrate_vector)).prod(axis=0) * 4
        term2 = np.sqrt(np.subtract(t1, t2))
        _velocities = np.vstack((np.subtract(term1, term2), np.divide(self.kcats, 2))).prod(axis=0)

        substrate_velocities[self.substrate_indices] = np.vstack((_velocities, self.substrate_stoichiometries)).prod(axis=0) * -1
        product_velocities[self.product_indices] = np.vstack((_velocities, self.product_stoichiometries)).prod(axis=0)
        return np.vstack((substrate_velocities, product_velocities)).sum(axis=0)
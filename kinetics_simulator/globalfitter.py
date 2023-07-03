import pandas as pd 
import numpy as np 
from scipy.optimize import minimize

class GlobalFitter:
    def __init__(self, model, variable_specie: str, observable_specie: str, fitting_params: list, fitting_concentrations: np.ndarray, ground_truth_data=np.ndarray, gaussian_noise=(0,0.01)):
        self.model = model
        self.variable_specie = variable_specie
        self.observable_specie, self.observable_index = observable_specie, self.model.species.index(observable_specie)
        self.fitting_params = fitting_params
        self.fitting_concentrations = fitting_concentrations
        self.ground_truth_data = ground_truth_data if type(ground_truth_data) != type else self._simulate_ground_truth_data(gaussian_noise)
        self.lookup = GlobalFitter._generate_lookup(self.fitting_concentrations)

        self.result = None 
        self.fits = np.ndarray
        self.residuals = np.ndarray
        self.fit_param_values = np.ndarray 
        self.chi_squared_data = pd.DataFrame 

    def _simulate_ground_truth_data(self, gaussian_noise: tuple):
        ground_truth_data = []
        for fitting_concen in self.fitting_concentrations:
            noise = np.random.normal(loc=gaussian_noise[0], scale=gaussian_noise[1], size=len(self.model.time)) if type(gaussian_noise) == tuple else np.zeros(len(self.model.time))
            self.model.update_dictionary[self.variable_specie](fitting_concen)
            ground_truth_data.append(self.model.integrate(self.model.initial_concentrations, self.model.time, inplace=False)[self.model.species.index(self.observable_specie)] + noise)
        return np.vstack(ground_truth_data)

    def _get_fitting_params(self):
        indices = np.array([])
        for param in self.fitting_params:
            mass_action_rates = np.array(self.model.mass_action_reactions.rate_names)
            Km_names = np.array(self.model.michaelis_menten_reactions.Km_names)
            kcat_names = np.array(self.model.michaelis_menten_reactions.kcat_names)

            mass_action_indices = np.argwhere(mass_action_rates == param).flatten()
            Km_indices = np.argwhere(Km_names == param).flatten()
            kcat_indices = np.argwhere(kcat_names == param).flatten()
            indices = np.hstack([indices, mass_action_indices, Km_indices, kcat_indices])
            if len(Km_indices) > 0 or len(kcat_indices) > 0:
                return 1
        return indices

    @staticmethod
    def _generate_lookup(fitting_concentrations: np.ndarray):
        lookup = {}
        for concen in set(fitting_concentrations):
            lookup[concen] = np.argwhere(fitting_concentrations == concen).flatten()
        return lookup

    def _simulate_data(self, x: np.ndarray, fitting_params: list):
        # update attributes for parameter attributes
        for param_value, param_name in zip(x, fitting_params):
            self.model.update_dictionary[param_name](param_value)

        observable_concentrations = np.zeros((len(self.fitting_concentrations), len(self.model.time)))

        # update initial concentrations of variable species and integrate
        for concentration in set(self.fitting_concentrations):
            self.model.update_dictionary[self.variable_specie](concentration)
            observable_concentrations[self.lookup[concentration]] = self.model.integrate(self.model.initial_concentrations, self.model.time, inplace=False)[self.observable_index]
        return observable_concentrations

    def compute_residuals(self, x: np.ndarray, fitting_params: list):
        observable_concentrations = self._simulate_data(x, fitting_params)
        residuals = np.square(observable_concentrations - self.ground_truth_data)
        return residuals

    def objective(self, x: np.ndarray, fitting_params: list):
        return self.compute_residuals(x, fitting_params).sum()

    def fit(self, fitting_params: list, inplace=True, fit_conversions=False):
        """
        Method for fitting a kinetic model to input or simulated data.
        """

        param_indices = self._get_fitting_params()
        if type(param_indices) == int:
            print('Fitting not supported for Michaelis Menten parameters in this version. This functionality will be included in a later version!')
            return
      
        K0 = np.ones(len(fitting_params) + 2) if fit_conversions else np.ones(len(fitting_params))
        bounds = [(0, None) for i in range(len(fitting_params))]
        result = minimize(self.objective, K0, bounds=bounds, args=(fitting_params,))

        if inplace:
            self.result = result
            self.fit_param_values = result.x
            self.residuals = self.compute_residuals(result.x, fitting_params)
            self.fits = self._simulate_data(result.x, fitting_params)
            return

        else:
            return result, result.x, self.compute_residuals(result.x, fitting_params), self._simulate_data(result.x, fitting_params)

    def generate_chi_squared_data(self, start_decrement=-5, end_decrement=5, no_points=12):
        chi_squared_arrays, grid_search_values = [], []
        for param, fit_param_value in zip(self.fitting_params, self.fit_param_values):
            chi_squared = []
            _grid_search_values = np.linspace(fit_param_value+start_decrement, fit_param_value+end_decrement, no_points)
            grid_search_values.append(_grid_search_values)
            for value in _grid_search_values:
                self.model.update_dictionary[param](value)
                fitting_params = [p for p in self.fitting_params if p != param]
                _, x = self.fit(fitting_params, inplace=False)

                # update model with fit parameters and re-integrate
                for v, p in zip(x, fitting_params):
                    self.model.update_dictionary[p](v)
                observable_timecourse = self.model.integrate(self.model.initial_concentrations, self.model.time, inplace=False)[self.observable_index]

                # compute chi squared statistic
                _chi_squared = np.square(observable_timecourse - self.ground_truth_data) / (len(self.ground_truth_data) - len(fitting_params))
                chi_squared.append(_chi_squared)
            chi_squared_arrays(np.array(chi_squared_arrays))
        
        self.chi_squared_data = dict([(self.fitting_params[i], (grid_search_values[i], chi_squared_arrays[i])) for i in range(len(self.fitting_params))])

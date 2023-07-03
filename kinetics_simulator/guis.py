""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for graphical user interfaces (guis).
"""

#imports 
import warnings
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import VBox, widgets

from .utils import SliderNameNotFoundWarning

class ProgressCurveGUI:
    def __init__(self, chemical_reaction_network, figsize=(8,8), title='Mass Action Kinetics', fontsize=12, multithread=False, tol=None, sliders=[]):
        self.chemical_reaction_network = chemical_reaction_network
        self.figsize, self.title, self.fontsize, self.multithread, self.tol, self.sliders = figsize, title, fontsize, multithread, tol, sliders
        self.fig = self._initialize_figure()

    def _get_data(self):
        self.chemical_reaction_network.integrate(self.chemical_reaction_network.initial_concentrations, self.chemical_reaction_network.time, rtol=self.tol, atol=self.tol)
        data = []
        for specie, concentration in zip(self.chemical_reaction_network.species, self.chemical_reaction_network.concentrations):
            data.append(dict(
                type='scatter',
                x=self.chemical_reaction_network.time,
                y=concentration,
                name=specie
            ))
        return data

    def _initialize_figure(self):
        data = self._get_data()
        fig = go.FigureWidget(data=data)
        fig.layout.title = self.title
        yaxis_text, xaxis_text = 'Concentration ({concen})', 'Time ({time})'
        fig.layout.yaxis.title = yaxis_text.format(concen=self.chemical_reaction_network.concentration_units)
        fig.layout.xaxis.title = xaxis_text.format(time=self.chemical_reaction_network.time_units)
        return fig

    def _generate_slider_update_function(self, name: str):
        multithreading = self.multithread

        def update_reaction_network(new_value):
            self.chemical_reaction_network.update_dictionary[name](new_value)
            self.chemical_reaction_network.integrate(self.chemical_reaction_network.initial_concentrations, self.chemical_reaction_network.time, rtol=self.tol, atol=self.tol)

        def update_figure():
            for specie_ind, concens in enumerate(self.chemical_reaction_network.concentrations): 
                with self.fig.batch_update():
                    self.fig.data[specie_ind].y = concens

        def slider_update(new_value):
            new_value = new_value['new']
            if multithreading:
                t1 = threading.Thread(target=update_reaction_network, args=(new_value,))
                t2 = threading.Thread(target=update_figure, args=())
                t1.start()
                t1.join()
                t2.start()
                t2.join()

            else:
                update_reaction_network(new_value)
                update_figure()

        return slider_update

    def _instantiate_sliders(self):
        slider_list = []
        for _slider in self.sliders:

            if _slider.name in self.chemical_reaction_network.species:
                start = self.chemical_reaction_network.initial_concentrations[self.chemical_reaction_network.species.index(_slider.name)]                
            elif _slider.name in self.chemical_reaction_network.mass_action_reactions.rate_names:
                ind = self.chemical_reaction_network.mass_action_reactions.rate_names.index(_slider.name)
                start = self.chemical_reaction_network.mass_action_reactions.K[ind, ind]
            elif _slider.name in self.chemical_reaction_network.michaelis_menten_reactions.Km_names:
                ind = self.chemical_reaction_network.michaelis_menten_reactions.Km_names.index(_slider.name)
                start = self.chemical_reaction_network.michaelis_menten_reactions.Kms[ind]
            elif _slider.name in self.chemical_reaction_network.michaelis_menten_reactions.kcat_names:
                ind = self.chemical_reaction_network.michaelis_menten_reactions.kcat_names.index(_slider.name)
                start = self.chemical_reaction_network.michaelis_menten_reactions.kcats[ind]
            else:
                warnings.warn(slider.name, SliderNameNotFoundWarning)
                continue

            if _slider.scale == 'log':
                slider = widgets.FloatLogSlider(
                    value=start,
                    min=_slider.min,
                    max=_slider.max,
                    step=_slider.stepsize,
                    continuous_update=_slider.continuous_update,
                    description=_slider.name
                )
                slider.observe(self._generate_slider_update_function(_slider.name), names='value')

            elif _slider.scale == 'linear':
                slider = widgets.FloatSlider(
                    value=start, 
                    base=_slider.base,
                    min=_slider.min,
                    max=_slider.max,
                    step=_slider.stepsize,
                    continuous_update=_slider.continuous_update,
                    description=_slider.name)
                slider.observe(self._generate_slider_update_function(_slider.name), names='value')

            slider_list.append(slider)
        return slider_list

    def interactive(self):
        sliders = self._instantiate_sliders()
        return VBox([self.fig] + sliders)

class BindingIsothermGUI(ProgressCurveGUI):
    def __init__(self, chemical_reaction_network, figsize=(8,8), title='Binding Kinetics', fontsize=12, multithread=False, tol=None, sliders=[]):
        self.chemical_reaction_network = chemical_reaction_network
        self.figsize, self.title, self.fontsize, self.multithread, self.tol, self.sliders = figsize, title, fontsize, multithread, tol, sliders
        self.fig = self._initialize_figure()

    def _get_data(self):
        self.chemical_reaction_network.get_progress_curves_and_isotherm()
        # self.chemical_reaction_network.fit_Kd()

        progress_curve_data, fraction_bound_data = [], []
        for L_concen, progress_curve in zip(self.chemical_reaction_network.ligand_concentrations, self.chemical_reaction_network.progress_curves):
            progress_curve_data.append(dict(
                type='scatter',
                x=self.chemical_reaction_network.time,
                y=progress_curve,
                name=f'{L_concen:+.3g}' + ' ' + self.chemical_reaction_network.concentration_units
            ))

        fraction_bound_data += [dict(
            type='scatter',
            x=self.chemical_reaction_network.ligand_concentrations,
            y=self.chemical_reaction_network.binding_isotherm,
            name='Simulated',
            mode='markers'
        )]
        fraction_bound_data += [dict(
            type='scatter',
            x=self.chemical_reaction_network.ligand_concentrations,
            y=self.chemical_reaction_network.ground_truth_binding_isotherm,
            name='Ground Truth',
            mode='markers'
        )]

        return progress_curve_data, fraction_bound_data

    def _initialize_figure(self):
        progress_curve_data, fraction_bound_data = self._get_data()
        Kd = self.chemical_reaction_network.mass_action_reactions.K[1,1] / self.chemical_reaction_network.mass_action_reactions.K[0,0]
        subs = make_subplots(cols=2, subplot_titles=['Progress Curves', 'Binding Isotherm'])
        fig = go.FigureWidget(subs)

        for data in progress_curve_data:
            fig.add_scatter(name=data['name'], x=data['x'], y=data['y'], row=1, col=1)
        for data in fraction_bound_data:
            fig.add_scatter(name=data['name'], x=data['x'], y=data['y'], row=1, col=2, mode=data['mode'], marker_size=9, marker_symbol='circle-open')
        fig.update_layout(title_text=self.title, title_x=0.5, title_font_size=28)
        fig['layout']['xaxis'].update(title_text='Time ' + '(' + self.chemical_reaction_network.time_units + ')')
        fig['layout']['yaxis'].update(title_text='[Complex] ' + '(' + self.chemical_reaction_network.concentration_units + ')')

        fig['layout']['xaxis2'].update(title_text='[Ligand] ' + '(' + self.chemical_reaction_network.concentration_units + ')')
        fig['layout']['yaxis2'].update(title_text='Fraction Bound')
        return fig 

    def _generate_slider_update_function(self, name: str):
        
        def update_reaction_network(new_value):
            self.chemical_reaction_network.update_dictionary[name](new_value)
            self.chemical_reaction_network.get_progress_curves_and_isotherm()
            self.chemical_reaction_network._get_ground_truth_binding_isotherm()
            # self.chemical_reaction_network.fit_Kd()

        def update_figure():
            for ind, curve in enumerate(self.chemical_reaction_network.progress_curves):
                with self.fig.batch_update():
                    self.fig.data[ind].y = curve

            self.fig.data[ind+1].y = self.chemical_reaction_network.binding_isotherm
            self.fig.data[ind+2].y = self.chemical_reaction_network.ground_truth_binding_isotherm

        def slider_update(new_value):
            new_value = new_value['new']
            if self.multithread:
                t1 = threading.Thread(target=update_reaction_network, args=(new_value,))
                t2 = threading.Thread(target=update_figure, args=())
                t1.start()
                t1.join()
                t2.start()
                t2.join()

            else:
                update_reaction_network(new_value)
                update_figure()

        return slider_update

    def _instantiate_sliders(self):
        return super()._instantiate_sliders()
    
    def interactive(self):
        return super().interactive()

class FitPlotGUI:
    def __init__(self, fitter, figsize=(8,8), title='Mass Action Kinetics', fontsize=12):
        self.fitter = fitter
        self.figsize, self.title, self.fontsize = figsize, title, fontsize
        self.residual_fig, self.base_title = self._initialize_residual_figure()
        self.slider = self._initialize_slider()

    def _initialize_residual_figure(self):
        subs = make_subplots(cols=2, subplot_titles=['Fits', 'Residuals'])
        fig = go.FigureWidget(subs)

        base_title = f'Progress Curves ([{self.fitter.observable_specie}]=' + '{:.2e}' + f'{self.fitter.model.concentration_units})\n' + \
                    ' '.join([param_name + ':' + f'{param:.2e}' for param_name, param in zip(self.fitter.fitting_params, self.fitter.fit_param_values)])

        fig.add_scatter(name='Ground Truth', x=self.fitter.model.time, y=self.fitter.ground_truth_data[0], row=1, col=1, mode='markers')
        fig.add_scatter(name='Fit', x=self.fitter.model.time, y=self.fitter.fits[0], row=1, col=1)
        fig.add_scatter(name='Residuals', x=self.fitter.model.time, y=self.fitter.residuals[0], row=1, col=2, mode='markers')

        fig.layout.title = base_title.format(self.fitter.fitting_concentrations[0])
        ylabel1 = '[{specie}] ({units})'
        fig['layout']['xaxis'].update(title_text='Time ' + '(' + self.fitter.model.time_units + ')')
        fig['layout']['yaxis'].update(title_text=ylabel1.format(specie=self.fitter.observable_specie, units=self.fitter.model.concentration_units))

        fig['layout']['xaxis2'].update(title_text='Time ' + '(' + self.fitter.model.time_units + ')')
        fig['layout']['yaxis2'].update(title_text='SSE')
        return fig, base_title

    def _generate_slider_update(self):
        def update(new_value):
            new_value = new_value['new']
            self.residual_fig.data[0].y = self.fitter.ground_truth_data[new_value]
            self.residual_fig.data[1].y = self.fitter.fits[new_value]
            self.residual_fig.data[2].y = self.fitter.residuals[new_value]
            self.residual_fig.layout.title = self.base_title.format(self.fitter.fitting_concentrations[new_value])
            return
        return update

    def _initialize_slider(self):
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.fitter.fitting_concentrations)-1,
            step=1,
            continuous_update=False,
            description='Index'
        )
        slider.observe(self._generate_slider_update(), 'value')
        return slider

    def launch_residual_figure(self):
        return VBox([self.residual_fig] + [self.slider])

class Slider:
    """
    Utility class for storing slider attributes.

    Attributes
    ----------
    name: str
        The name of the rate constant or specie for the slider.
    min: float
        Minimum value for slider. If logscale, corresponds to 
        the minimum exponent.
    max: float
        Maximum value for slider. If logscale, corresponds to 
        the maximum exponent.
    stepsize: float
        Slider stepsize.
    scale: str  
        Can take on 'log' or 'linear'. 
    base: float
        If the scale is 'log', defines base of exponential.
    continuous_update: bool
        Determines if plots are continuously updated as slider
        is moved. Should be set to False if dealing with complex
        kinetic mechanisms.
    """
    def __init__(self, name: str, min=0, max=100, stepsize=1, scale='log', base=10, continuous_update=True):
        self.name, self.min, self.max, self.stepsize, self.scale, self.base, self.continuous_update = name, min, max, stepsize, scale, base, continuous_update
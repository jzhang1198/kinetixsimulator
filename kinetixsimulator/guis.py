""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for graphical user interfaces.
"""

#imports 
import threading
import numpy as np
from math import ceil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .chemicalkinetics import KineticModel
from ipywidgets import VBox, HBox, widgets

class ProgressCurveGUI:
    def __init__(
            self, 
            figsize=(8,8), 
            title='Mass Action Kinetics', 
            fontsize=12,
            multithread=False, 
            ):
        
        self.figsize = figsize
        self.title = title
        self.fontsize = fontsize
        self.multithread = multithread
    
    def _init_figure(self, kinetic_model: KineticModel, hidden_species: set):

        # integrate reaction network and get data
        kinetic_model.simulate()
        data = []
        for specie, concentration in zip(kinetic_model.species, kinetic_model.simulated_data):

            data_dict = dict(
                type='scatter',
                x=kinetic_model.time,
                y=concentration,
                name=specie,
            )
            
            if specie in hidden_species:
                data_dict['visible'] = 'legendonly'

            data.append(data_dict)

        fig = go.FigureWidget(data=data)
        fig.layout.title = self.title
        yaxis_text, xaxis_text = 'Concentration ({concen})', 'Time ({time})'
        fig.layout.yaxis.title = yaxis_text.format(concen=kinetic_model.concentration_units)
        fig.layout.xaxis.title = xaxis_text.format(time=kinetic_model.time_units)
        return fig

    def _generate_slider_update_function(self, name: str, kinetic_model: KineticModel, fig):
        multithreading = self.multithread

        def update_kinetic_model(new_value):
            kinetic_model.update_dictionary[name](new_value)
            kinetic_model.simulate()

        def update_figure():
            for specie_ind, simulated_curve in enumerate(kinetic_model.simulated_data): 
                with fig.batch_update():
                    fig.data[specie_ind].y = simulated_curve

        def slider_update(new_value):
            new_value = new_value['new']
            if multithreading:
                t1 = threading.Thread(target=update_kinetic_model, args=(new_value,))
                t2 = threading.Thread(target=update_figure, args=())
                t1.start()
                t1.join()
                t2.start()
                t2.join()

            else:
                update_kinetic_model(new_value)
                update_figure()

        return slider_update

    def _init_sliders(self, kinetic_model: KineticModel, fig, slider_config: dict):

        # set default slider params
        n_steps_def = 1000
        slider_list, log_slider_list = [], []

        specie_lb_def, specie_ub_def = 0, 1e3
        for name in kinetic_model.species:

            if name in slider_config.keys() and not slider_config[name]:
                continue

            index = kinetic_model.species.index(name)
            value = kinetic_model.specie_initial_concs[index]
            lb = specie_lb_def if name not in slider_config.keys() else slider_config[name][0]
            ub = specie_ub_def if name not in slider_config.keys() else slider_config[name][1]

            stepsize = (ub - lb) / (n_steps_def - 1)
            log_lb = -10 if lb == 0 else np.log10(lb)
            log_ub = -10 if ub == 0 else np.log10(ub)
            log_value = 1e-10 if value == 0 else value
            log_stepsize = (log_ub - log_lb) / (n_steps_def - 1)

            # make sliders 
            slider = widgets.FloatSlider(
                value=value,
                min=lb,
                max=ub,
                step=stepsize,
                continuous_update=True,
                description=name
            )
            slider.observe(self._generate_slider_update_function(name, kinetic_model, fig), names='value')
            log_slider = widgets.FloatLogSlider(
                value=log_value,
                min=log_lb,
                max=log_ub,
                base=10,
                step=log_stepsize,
                continuous_update=True,
                description=name
            )
            log_slider.observe(self._generate_slider_update_function(name, kinetic_model, fig), names='value')
            slider_list.append(slider), log_slider_list.append(log_slider)

        rconst_lb_def, rconst_ub_def = 1e-8, 1e8
        for name in kinetic_model.rconst_names:

            if name in slider_config.keys() and not slider_config[name]:
                continue

            index = kinetic_model.rconst_names.index(name)
            value = kinetic_model.rconst_values[index]
            lb = rconst_lb_def if name not in slider_config.keys() else slider_config[name][0]
            ub = rconst_ub_def if name not in slider_config.keys() else slider_config[name][1]

            stepsize = (ub - lb) / (n_steps_def - 1)
            log_lb = -10 if lb == 0 else np.log10(lb)
            log_ub = -10 if ub == 0 else np.log10(ub)
            log_value = 1e-10 if value == 0 else value
            log_stepsize = (log_ub - log_lb) / (n_steps_def - 1)

            # make sliders 
            slider = widgets.FloatSlider(
                value=value,
                min=lb,
                max=ub,
                step=stepsize,
                continuous_update=True,
                description=name
            )
            slider.observe(self._generate_slider_update_function(name, kinetic_model, fig), names='value')
            log_slider = widgets.FloatLogSlider(
                value=log_value,
                min=log_lb,
                max=log_ub,
                base=10,
                step=log_stepsize,
                continuous_update=True,
                description=name
            )
            log_slider.observe(self._generate_slider_update_function(name, kinetic_model, fig), names='value')
            slider_list.append(slider), log_slider_list.append(log_slider)

        MM_lb_def, MM_ub_def = 1e-6, 1e6
        for name in kinetic_model.MM_rconst_names:

            if name in slider_config.keys() and not slider_config[name]:
                continue

            index = kinetic_model.MM_rconst_names.index(name)
            value = kinetic_model.MM_rconst_values[index]
            lb = MM_lb_def if name not in slider_config.keys() else slider_config[name][0]
            ub = MM_ub_def if name not in slider_config.keys() else slider_config[name][1]

            stepsize = (ub - lb) / (n_steps_def - 1)
            log_lb = -10 if lb == 0 else np.log10(lb)
            log_ub = -10 if ub == 0 else np.log10(ub)
            log_value = 1e-10 if value == 0 else value
            log_stepsize = (log_ub - log_lb) / (n_steps_def - 1)

            # make sliders 
            slider = widgets.FloatSlider(
                value=value,
                min=lb,
                max=ub,
                step=stepsize,
                continuous_update=True,
                description=name
            )
            slider.observe(self._generate_slider_update_function(name, kinetic_model, fig), names='value')
            log_slider = widgets.FloatLogSlider(
                value=log_value,
                min=log_lb,
                max=log_ub,
                base=10,
                step=log_stepsize,
                continuous_update=True,
                description=name
            )
            log_slider.observe(self._generate_slider_update_function(name, kinetic_model, fig), names='value')
            slider_list.append(slider), log_slider_list.append(log_slider)

        return slider_list, log_slider_list
    
    def _init_toggle_buttons(self, sliders: list, log_sliders: list, n_slider_cols: int):
        
        hbox_container = dict([i, []] for i in range(1, n_slider_cols + 1))
        no_sliders_per_container = ceil(len(sliders) / n_slider_cols)
        slider_dict = {}  # Dictionary to keep track of active sliders for each pair

        incrementor = 0
        current_vbox_container = 1
        for slider, log_slider in zip(sliders, log_sliders):
            # Initialize the active slider as the linear slider
            slider_dict[slider.description] = slider
            
            toggle_button = widgets.Button(description='Linear Scale')
            toggle_button.layout.width = '100px'  # Adjust the width as needed
            toggle_button.layout.height = '25px'   # Adjust the height as needed

            # Function to toggle between linear and log sliders
            def toggle_scale(change, slider, log_slider, toggle_button, slider_container):
                current_description = toggle_button.description
                if current_description == 'Linear Scale':
                    toggle_button.description = 'Log Scale'
                    slider_dict[slider.description] = log_slider  # Update the active slider
                    log_slider.value = slider.value
                else:
                    toggle_button.description = 'Linear Scale'
                    slider_dict[slider.description] = slider  # Update the active slider
                    slider.value = log_slider.value
                # Update the HBox container with the new active slider
                slider_container.children = [toggle_button, slider_dict[slider.description]]

            slider_container = HBox([toggle_button, slider_dict[slider.description]])
            toggle_button.on_click(lambda change, s=slider, ls=log_slider, tb=toggle_button, slider_container=slider_container: toggle_scale(change, s, ls, tb, slider_container))
            
            if incrementor == no_sliders_per_container:
                incrementor = 0
                current_vbox_container += 1

            hbox_container[current_vbox_container].append(slider_container)
            incrementor += 1

        return HBox([VBox(i) for i in hbox_container.values()])

    def launch(self, kinetic_model: KineticModel, slider_config: dict = {}, hidden_species: list = [], n_slider_cols: int = 3):
        fig = self._init_figure(kinetic_model, set(hidden_species))
        sliders, log_sliders = self._init_sliders(kinetic_model, fig, slider_config)
        slider_containers = self._init_toggle_buttons(sliders, log_sliders, n_slider_cols)
        return VBox([fig] + [slider_containers])

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
        Kd = self.chemical_reaction_network.reaction_collection.K[1,1] / self.chemical_reaction_network.reaction_collection.K[0,0]
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
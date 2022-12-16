""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for graphical user interfaces (guis).
"""

#imports 
import threading
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go
from ipywidgets import VBox, HBox, widgets

class ReactionField(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.fields = []
        self.createMMButton()
        self.createMassActionButton()
        self.grid(column=0, sticky="NEWS")
        self.add_MM, self.add_mass_action = None, None

    def createMMButton(self):        
        self.add_field()
        self.add_MM = ttk.Button(self, text="Add MM Reaction", command=self.add_MM)
        self.add_MM.bind("<Return>", self.add_field)
        self.add_MM.grid(row=len(self.fields), column=3, padx=4, pady=6, sticky="W")

    def createMassActionButton(self):
        self.add_field()
        self.add_mass_action = ttk.Button(self, text="Add MM Reaction", command=self.add_field)
        self.add_mass_action.bind("<Return>", self.add_field)
        self.add_mass_action.grid(row=len(self.fields), column=8, padx=4, pady=6, sticky="W")

    def add_MM_field(self):
        self.fields.append({})
        n = len(self.fields)-1
        self.fields[n]['var'] = tk.StringVar(self)
        self.fields[n]['field'] = ttk.Entry(self, textvariable=self.fields[n]['var'])
        self.fields[n]['field'].grid(row=n, column=0, columnspan=2, padx=4, pady=6, sticky="NEWS")
        if n:
            self.add_lang.grid(row=n + 1, column=3, padx=4, pady=6, sticky="W")

    def add_mass_action_field(self):
        pass

class ProgressCurveGUI:
    def __init__(self, chemical_reaction_network, plot_kwargs: dict):
        self.chemical_reaction_network = chemical_reaction_network
        self.plot_kwargs = ProgressCurveGUI._process_plot_kwargs(plot_kwargs)
        self.fig = self._initialize_figure()

    @staticmethod
    def _process_plot_kwargs(plot_kwargs: dict):
        plot_kwargs['figsize'] = (8,8) if 'figsize' not in plot_kwargs.keys() else plot_kwargs['figsize']
        plot_kwargs['title'] = 'Mass Action Kinetics' if 'title' not in plot_kwargs.keys() else plot_kwargs['title']
        plot_kwargs['fontsize'] = 12 if 'fontsize' not in plot_kwargs.keys() else plot_kwargs['fontsize']
        plot_kwargs['multithread'] = False if 'multithread' not in plot_kwargs.keys() else plot_kwargs['multithread']
        plot_kwargs['tol'] = None if 'tol' not in plot_kwargs.keys() else plot_kwargs['tol']
        return plot_kwargs

    def _get_data(self):
        self.chemical_reaction_network.integrate(self.chemical_reaction_network.initial_concentrations, self.chemical_reaction_network.time, rtol=self.plot_kwargs['tol'], atol=self.plot_kwargs['tol'])
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
        fig.layout.title = self.plot_kwargs['title']
        yaxis_text, xaxis_text = 'Concentration ({concen})', 'Time ({time})'
        fig.layout.yaxis.title = yaxis_text.format(concen=self.chemical_reaction_network.concentration_units)
        fig.layout.xaxis.title = xaxis_text.format(time=self.chemical_reaction_network.time_units)
        return fig

    def _generate_slider_update_function(self, name: str):
        multithreading = self.plot_kwargs['multithread']

        def update_reaction_network(new_value):
            self.chemical_reaction_network.update_dictionary[name](new_value)
            self.chemical_reaction_network.integrate(self.chemical_reaction_network.initial_concentrations, self.chemical_reaction_network.time, rtol=self.plot_kwargs['tol'], atol=self.plot_kwargs['tol'])

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
        slider_names = set(self.plot_kwargs.keys()).intersection(set(self.chemical_reaction_network.update_dictionary.keys()))
        sliders = []
        for name in slider_names:
            slider = widgets.FloatSlider(
                value=self.plot_kwargs[name]['start'], 
                min=self.plot_kwargs[name]['min'],
                max=self.plot_kwargs[name]['max'],
                step=self.plot_kwargs[name]['stepsize'],
                continuous_update=True,
                description=name)
            slider.observe(self._generate_slider_update_function(name), names='value')
            sliders.append(slider)
        return sliders

    def interactive(self):
        sliders = self._instantiate_sliders()
        return VBox([self.fig] + sliders)

class MMCurveGUI:
    def __init__(self, coupled_enzyme_network, plot_kwargs: dict):
        self.chemical_reaction_network = coupled_enzyme_network
        self.plot_kwargs = MMCurveGUI._process_plot_kwargs(plot_kwargs)
        self.MM_fig, self.d1_fig, self.d2_fig, self.d3_fig, self.progress_curve_fig = self._initialize_figure()

    @staticmethod
    def _process_plot_kwargs(plot_kwargs: dict):
        plot_kwargs['figsize'] = (8,8) if 'figsize' not in plot_kwargs.keys() else plot_kwargs['figsize']
        plot_kwargs['title'] = 'Mass Action Kinetics' if 'title' not in plot_kwargs.keys() else plot_kwargs['title']
        plot_kwargs['fontsize'] = 12 if 'fontsize' not in plot_kwargs.keys() else plot_kwargs['fontsize']
        plot_kwargs['multithread'] = False if 'multithread' not in plot_kwargs.keys() else plot_kwargs['multithread']
        plot_kwargs['tol'] = None if 'tol' not in plot_kwargs.keys() else plot_kwargs['tol']
        plot_kwargs['compute_deriv'] = False if 'compute_deriv' not in plot_kwargs.keys() else plot_kwargs['compute_deriv']
        plot_kwargs['progress_curves'] = False if 'progress_curves' not in plot_kwargs.keys() else plot_kwargs['progress_curves']
        plot_kwargs['continuous_update'] = False if 'continuous_update' not in plot_kwargs.keys() else plot_kwargs['continuous_update']
        return plot_kwargs
 
    def _initialize_figure(self):
        MM_data, d1_data, d2_data, d3_data, progress_curve_data = self._get_data()
        d1_fig, d2_fig, d3_fig, progress_curve_fig = [None] * 4

        MM_fig = go.FigureWidget(data=MM_data)
        title_text = '{title}\nk<sub>cat</sub> = {kcat:.1f}, K<sub>m</sub> = {Km:.1f}'
        ylabel_text = 'V<sub>0</sub> ({conc_units}/{time_units})'
        xlabel_text = '[S] ({conc_units})'
        MM_fig.layout.title = title_text.format(title=self.plot_kwargs['title'], kcat=self.chemical_reaction_network.fit_params[0], Km=self.chemical_reaction_network.fit_params[1])
        MM_fig.layout.yaxis.title = ylabel_text.format(conc_units=self.chemical_reaction_network.concentration_units, time_units=self.chemical_reaction_network.time_units)
        MM_fig.layout.xaxis.title = xlabel_text.format(conc_units=self.chemical_reaction_network.concentration_units)
        if self.plot_kwargs['compute_deriv']:
            d1_fig = go.FigureWidget(data=d1_data)
            d2_fig = go.FigureWidget(data=d2_data)
            d3_fig = go.FigureWidget(data=d3_data)

        if self.plot_kwargs['progress_curves']:
            progress_curve_fig = go.FigureWidget(data=progress_curve_data)

        return MM_fig, d1_fig, d2_fig, d3_fig, progress_curve_fig

    def _get_data(self):
        self.chemical_reaction_network.simulate_mm_model_fit(compute_deriv=self.plot_kwargs['compute_deriv'], progress_curves=self.plot_kwargs['progress_curves'])
        gt_params, fit_params = self.chemical_reaction_network.ground_truth_params, self.chemical_reaction_network.fit_params
        E = self.chemical_reaction_network.initial_concentrations[self.chemical_reaction_network.michaelis_menten_reactions.enzyme_indices[0]]

        d1_data, d2_data, d3_data, progress_curve_data = [], [], [], []
        MM_data = [
            dict(
                type='scatter', 
                x=self.chemical_reaction_network.concentration_range,
                y=self.chemical_reaction_network._mm_model(self.chemical_reaction_network.concentration_range, gt_params[0], gt_params[1]),
                name='Ground Truth'
                ),
            dict(
                type='scatter', 
                x=self.chemical_reaction_network.concentration_range,
                y=self.chemical_reaction_network._mm_model(self.chemical_reaction_network.concentration_range, fit_params[0], fit_params[1]),
                name='Fit'       
            )
        ]

        if self.plot_kwargs['compute_deriv']:
            for i in range(len(self.chemical_reaction_network.substrate_concentrations)):
                S = self.chemical_reaction_network.substrate_concentrations[i]
                d1_data.append(
                    dict(
                        type='scatter',
                        x=self.chemical_reaction_network.timecourse,
                        y=self.chemical_reaction_network.d1[:,i],
                        name=f'{S}'
                    )
                )
                d2_data.append(
                    dict(
                        type='scatter',
                        x=self.chemical_reaction_network.timecourse,
                        y=self.chemical_reaction_network.d2[:,i],
                        name=f'{S}'
                    )
                )
                d3_data.append(
                    dict(
                        type='scatter',
                        x=self.chemical_reaction_network.timecourse,
                        y=self.chemical_reaction_network.d3[:i],
                        name=f'{S}'
                    )
                )
            
        if self.plot_kwargs['progress_curves']:
            for i in range(len(self.chemical_reaction_network.substrate_concentrations)):
                S = self.chemical_reaction_network.substrate_concentrations[i]
                progress_curve_data.append(
                    dict(
                        type='scatter',
                        x=self.chemical_reaction_network.timecourse,
                        y=self.chemical_reaction_network.progress_curves[:,i],
                        name=f'{S}'
                    )
                )
                
        return MM_data, d1_data, d2_data, d3_data, progress_curve_data
    
    def _generate_slider_update_function(self, name: str):
        multithreading = self.plot_kwargs['multithread']
        title_text = '{title}\nk<sub>cat</sub> = {kcat:.1f}, K<sub>m</sub> = {Km:.1f}'

        def update_reaction_network(new_value):
            self.chemical_reaction_network.update_dictionary[name](new_value)
            self.chemical_reaction_network.simulate_mm_model_fit(compute_deriv=self.plot_kwargs['compute_deriv'], progress_curves=self.plot_kwargs['progress_curves'])

        def update_figure():
            gt_params, fit_params = self.chemical_reaction_network.ground_truth_params, self.chemical_reaction_network.fit_params
            self.MM_fig.data[0].y = self.chemical_reaction_network._mm_model(self.chemical_reaction_network.concentration_range, gt_params[0], gt_params[1])
            self.MM_fig.data[1].y = self.chemical_reaction_network._mm_model(self.chemical_reaction_network.concentration_range, fit_params[0], fit_params[1])
            self.MM_fig.layout.title = title_text.format(title=self.plot_kwargs['title'], kcat=self.chemical_reaction_network.fit_params[0], Km=self.chemical_reaction_network.fit_params[1])

            if self.plot_kwargs['compute_deriv']:
                for i in range(len(self.chemical_reaction_network.substrate_concentrations)):
                    self.d1_fig.data[i].x, self.d2_fig.data[i].x, self.d3_fig.data[i].x = [self.chemical_reaction_network.timecourse] * 3
                    self.d1_fig.data[i].y = self.chemical_reaction_network.d1[:,i]
                    self.d2_fig.data[i].y = self.chemical_reaction_network.d2[:,i]
                    self.d3_fig.data[i].y = self.chemical_reaction_network.d3[:,i]

            if self.plot_kwargs['progress_curves']:
                for i in range(len(self.chemical_reaction_network.substrate_concentrations)):
                    self.progress_curve_fig.data[i].x= self.chemical_reaction_network.timecourse
                    self.progress_curve_fig.data[i].y = self.chemical_reaction_network.progress_curves[:,i]

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
        slider_names = set(self.plot_kwargs.keys()).intersection(set(self.chemical_reaction_network.update_dictionary.keys()))
        sliders = []
        for name in slider_names:
            slider = widgets.FloatSlider(
                value=self.plot_kwargs[name]['start'], 
                min=self.plot_kwargs[name]['min'],
                max=self.plot_kwargs[name]['max'],
                step=self.plot_kwargs[name]['stepsize'],
                continuous_update=self.plot_kwargs['continuous_update'],
                description=name)
            slider.observe(self._generate_slider_update_function(name), names='value')
            sliders.append(slider)
        return sliders

    def interactive(self):
        sliders = self._instantiate_sliders()

        if self.plot_kwargs['compute_deriv']:
            fig = VBox([HBox([self.MM_fig, self.d1_fig, self.d2_fig, self.d3_fig]), sliders])

        elif self.plot_kwargs['progress_curves']:
            pafig = VBox([HBox([self.MM_fig, self.progress_curve_fig]), sliders])

        elif self.plot_kwargs['compute_deriv'] and self.plot_kwargs['progress_curves']:
            fig = VBox([HBox([self.MM_fig, self.d1_fig, self.d2_fig, self.d3_fig, self.progress_curve_fig]), sliders])

        else:
            fig = VBox([self.MM_fig] + sliders)

        return fig

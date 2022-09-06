""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for graphical user interfaces (guis).
"""

#imports 
import threading
import plotly.graph_objects as go
from ipywidgets import VBox, widgets

class PlotGUI:
    def __init__(self, chemical_reaction_network, plot_kwargs: dict):
        self.chemical_reaction_network = chemical_reaction_network
        self.plot_kwargs = PlotGUI._process_plot_kwargs(plot_kwargs)
        self.fig = self._initialize_figure()

    @staticmethod
    def _process_plot_kwargs(plot_kwargs: dict):
        plot_kwargs['figsize'] = (8,8) if 'figsize' not in plot_kwargs.keys() else plot_kwargs['figsize']
        plot_kwargs['title'] = 'Mass Action Kinetics' if 'title' not in plot_kwargs.keys() else plot_kwargs['title']
        plot_kwargs['xlabel'] = 'Time (s)' if 'xlabel' not in plot_kwargs.keys() else plot_kwargs['xlabel']
        plot_kwargs['ylabel'] = 'Concentration (nM)' if 'ylabel' not in plot_kwargs.keys() else plot_kwargs['ylabel']
        plot_kwargs['fontsize'] = 12 if 'fontsize' not in plot_kwargs.keys() else plot_kwargs['fontsize']
        plot_kwargs['multithread'] = False if 'multithread' not in plot_kwargs.keys() else plot_kwargs['multithread']
        plot_kwargs['tol'] = None if 'tol' not in plot_kwargs.keys() else plot_kwargs['tol']
        return plot_kwargs

    def _get_data(self):
        self.chemical_reaction_network.integrate(rtol=self.plot_kwargs['tol'], atol=self.plot_kwargs['tol'])
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
        fig.layout.yaxis.title = self.plot_kwargs['ylabel']
        fig.layout.xaxis.title = self.plot_kwargs['xlabel']
        return fig

    def interactive(self):
        sliders = self._instantiate_sliders()
        return VBox([self.fig] + sliders)

    def _generate_slider_update_function(self, name: str):
        multithreading = self.plot_kwargs['multithread']

        def update_reaction_network(new_value):
            self.chemical_reaction_network.update_dictionary[name](new_value)
            self.chemical_reaction_network.integrate(rtol=self.plot_kwargs['tol'], atol=self.plot_kwargs['tol'])

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

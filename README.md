# kinetixsimulator

A Python package for simulating chemical reaction kinetics and global fitting of kinetic data. 

## Before You Begin

Ensure you have [git](https://git-scm.com/downloads) installed. A package manager (like [anaconda](https://docs.conda.io/projects/conda/en/latest/index.html)) is highly recommended. Within your virtual environment, you will need to install `python==3.9.12`. I've experienced issues with installation if this step is not done beforehand.

## Installation

After you clone the repo to your local device, run `pip install -e .` to install the library and its dependencies.

## Usage

### Constructing a KineticModel

The `KineticModel` class is a Python implementation designed for modeling and simulating chemical kinetic mechanisms. A `KineticModel` object can be constructed as follows:

```python
# import dependencies
from kinetixsimulator.chemicalkinetics import KineticModel, Reaction, ReversibleReaction
from matplotlib import pyplot as plt
import numpy as np

# create a list of reaction objects
reactions = [
    ReversibleReaction(reaction_string='E + S <-> E:S', rconst_names=['kon', 'koff'], rconst_values=[1e2, 1]),
    Reaction(reaction_string='E:S -> E + P', rconst_name='kcat', rconst_value=5)
]
```

When constructing reacting strings, keep the following formatting rules in mind:
1. Separate chemical species with a `+`
2. Separate substrates and products with a `->` or `<->`
3. Separate stoichiometric coefficients with a `*`
4. Separate species and characters with a space

If you are modelling kinetics of an enzymatic reaction, you may consider using the `MMReaction` container class. Refer to [Technical Notes](#technical-notes) for a more detailed discussion of usage of this class.

### Initializing a KineticModel

By default, initial concentrations of all species are set to zero. Initial concentrations can be manually set using the `set_initial_concentrations` method provided within the `KineticModel` class. Note that the units of concentration default to µM. This can be overriden with an optional argument during construction of a `KineticModel` object.

```python
kinetic_model.set_initial_concentration('E', 1e-3)
kinetic_model.set_initial_concentration('S', 2)
```

### Simulating Reaction Kinetics

Reaction kinetics under a given set of initial concentrations and rate constants can be simulated using the `simulate` method within the `KineticModel` class.

```python
# simulate and plot the results
fig, ax = plt.subplots()
traces = kinetic_model.simulate(inplace=False)
for specie_name, trace in zip(kinetic_model.species, traces):
    ax.plot(kinetic_model.time, trace, label=specie_name)

ax.set_xlabel(f'Time ({kinetic_model.time_units})')
ax.set_ylabel(f'Concentration ({kinetic_model.concentration_units})')
ax.legend()
```

### Interactive Simulation of Reaction Kinetics

To launch the interactive simulation dashboard, construct a `ProgressCurveGUI` object and call the `launch` method.

```python
from kinetixsimulator.guis import ProgressCurveGUI

gui = ProgressCurveGUI()
gui.launch(kinetic_model, hidden_species=['E:S'], slider_config={'E:S': None, 'P': None, 'S': (1, 100)})
```

By default, sliders enabling control of rate constant values and specie initial concentrations will be rendered with a set range. You can prevent the display of particular sliders and override default slider ranges by passing a `slider_config` dictionary to the `launch` method. If you would like to hide traces for certain chemical species by providing a `hidden_species` list to the `launch` method. Note that interactive simulation is only supported within the Jupyter environment (i.e. you can't run this outside of a Jupyter Notebook).

### Technical Notes

#### Avoiding Un-Physical Solutions

If your model contains rate constants or initial concentrations that are exceptionally large, you may observe un-physical solutions (i.e. negative concentrations for some species). This can be avoided by lowering the values of the `integrator_atol` and `integrator_rtol` optional arguments in the `KineticModel` constructor, which set the error tolerance of the numerical integration algorithm used in the `simulate` method. Decreasing the increment between timepoints may help resolve these issues.

#### Using the MMReaction Class

The `MMReaction` class provides a more convenient means to organize data for enzymatic reactions that obey the single substrate Michaelis-Menten equation. A `MMReaction` object can be constructed as follows:

```python
reaction = MMReaction(reaction_string='E + S <-> E:S -> E + P', Km_name='Km', Km_value=1, kcat_name='kcat', kcat_value=1)
```

The reaction strings used to construct a `MMReaction` object must contain an equilibrium describing the substrate binding step and an irreversible reaction corresponding to the chemical step. Currently, the class only supports single substrate Michaelis-Menten mechanisms. If your enzyme follows a multi substrate Michaelis-Menten mechanism, you might consider simulating your system under conditions enabling simplification to a single substrate mechanism. 

A word of caution: during construction of the `KineticModel` object, values for microscopic rate constants (i.e. substrate association and dissociation rate constants) that are consistent with the provided steady-state constants will be selected arbitrarily. Keep this in mind when interpreting simulation results! 
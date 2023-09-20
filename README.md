# Kinetics Simulator

This notebook allows you to simulate the kinetics of a chemical reaction network.

## Before You Begin

Ensure you have [git](https://git-scm.com/downloads) installed. A package manager (like [anaconda](https://docs.conda.io/projects/conda/en/latest/index.html)) is highly recommended. Within your virtual environment, you will need to install `python==3.9.12`. I've experienced issues with installation if this step is not done beforehand.

## Installation

After you clone the repo to your local device, run `pip install -e .` to install the library and its dependencies.


## Theory

As enzymologists, we are often interested in simulating the behaviour of a system of coupled enzymatic reactions. If we have prior knowledge of the rate constants associated with these individual reactions, we can model their behaviours using mass action or Michaelis Menten kinetics. For a chemical reaction network, this is done by first defining a vector of rates, where each element corresponds to the time derivative of a chemical species. We can always construct such a vector using chemical rate laws and the Michaelis Menten equation. After computing this rate vector, we can numerically integrate to solve for a concentration vector for each timepoint. 

## Running the Notebook

To run a kinetic simulation, you will need to define an array of timepoints and three dictionaries. The first dictionary, <code>mass_action_dict</code>, contains all the elementary reactions modeled using mass action kinetics and their corresponding rate constants. The enzymatic reactions modelled by Michaelis-Menten kinetics are contained in the <code>michaelis_menten_dict</code>. The third and final dictionary, <code>initial_values</code>, defines the initial state of the chemical reaction network.

When writing these dictionaries, ensure you abide by the following conventions:
1) Substrates names and rate constant names should have no spaces and none of the following special characters: *, ->, +, <->. Note that all names should be unique.
2) All substrates, characters, and stoichiometric coefficients should be separated by a single space.
3) Reversible reactions are designated by a '<->' character. The two rate constants are associated with the forward and reverse reactions respectively.
4) Initial substrate concentrations left undefined will automatically be set to zero.
5) If you have no Michaelis Menten or mass action reactions, it is ok to leave their respective dictionaries empty.
6) The <code>plot_kwargs</code> dictionary is completely optional. If you would like more control over the appearance of your plot, feel free to add key-value pairs as appropriate.

## Known Issues

The simulation algorithm may produce un-physical solutions for systems with very fast kinetics. Be careful when assigning large values to rate constants. I am actively working on implementing changes to enable better handling of these systems.

I've also noticed that, in rare cases, the plot itself will not render. I believe this is an issue with ipywidgets, but I'm not entirely sure what the exact issue is. Usually, a combination of kernel restarts and re-running cells will resolve the issue.
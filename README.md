[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![build workflow](https://github.com/AlexanderSouthan/pyExperimentalDesign/actions/workflows/main.yml/badge.svg)](https://github.com/AlexanderSouthan/pyExperimentalDesign/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/AlexanderSouthan/pyExperimentalDesign/branch/master/graph/badge.svg?token=E17XE4FR6H)](https://codecov.io/gh/AlexanderSouthan/pyExperimentalDesign)

# pyExperimentalDesign

Class for the analysis of data obtained from a design of experiments approach. It does aim to re-invent the relevant, individual methods already implemented in SciPy, statsmodels, scikit-learn etc., but rather to provide an easy to use module that does some data preprocessing and is a collection of the most useful methods from the external packages. 

* Scales the parameter ranges to the interval of [-1, 1]. Thus, the fit parameters of the regression model can be used directly to compare the effects of the various experimental parameters. 
* Does the regression with linear, two-factor and three-factor interaction as well as quadratic models.
* Performs an analysis of variance (ANOVA) on the data.
* Also contains some useful methods for generation of plots for model diagnostics.

## Installation
Download and run the following command from the repository folder works:
```
pip install -e .
```

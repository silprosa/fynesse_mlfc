# Socio-Economic and Spatial Factors in Kenyan Education


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This project investigates the socio-economic and spatial factors that influence educational access and attainment in Kenya. Using household- and individual-level data from the Kenyan National Household Survey, the analysis follows the Access–Assess–Address framework. The study examines disparities in school attendance and attainment across demographic groups, the role of household socio-economic status, and infrastructural constraints such as distance to facilities. Findings provide evidence to guide education policy and interventions aimed at improving equity in learning opportunities.



## Project Structure
```
fynesse/
├── access.py      # Data access functionality
├── assess.py      # Data assessment and quality checks
├── address.py     # Question addressing and analysis
├── config.py      # Configuration management
├── defaults.yml   # Default configuration values
└── tests/         # Comprehensive test suite
    ├── test_access.py
    ├── test_assess.py
    └── test_address.py
```




## Installation
Clone or fork the repository then

```
!git clone https://github.com/your-username/fynesse_mlfc.git
import os, subprocess, importlib, sys
sys.path.append("/content/fynesse_mlfc")

```
---
The analysis employs the Access–Assess–Address framework:

Access – Explore school attendance patterns across age, gender, county, and household status.

Assess – Investigate how socio-economic factors (income, parental education, household head status) shape attainment and participation.

Address – Identify spatial and infrastructural barriers (distance to schools, electricity, internet, road access) that constrain educational opportunities.



## Usage
- *`access.py`*: Implement the `data()` function to load ydata sources
- *`assess.py`*: plot(`your_plotting_function()`, `plot_counties()`, `view()`, `labelled()`)
- *`address.py`*: Implement analysis and question-addressing functionality




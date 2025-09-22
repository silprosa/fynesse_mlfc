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

---Access--- – Explore school attendance patterns across age, gender, county, and household status.

Assess – Investigate how socio-economic factors (income, parental education, household head status) shape attainment and participation.

Address – Identify spatial and infrastructural barriers (distance to schools, electricity, internet, road access) that constrain educational opportunities.



## Using the Framework

### Template Implementation
The framework is provided as a template with stub implementations. Each module contains:

- *`access.py`*: Implement the `data()` function to load ydata sources
- *`assess.py`*: plot(`your_plotting_function()`, `plot_counties()`, `view()`, `labelled()`)
- *`address.py`*: Implement analysis and question-addressing functionality

### Error Handling and Logging

The framework includes basic error handling and logging to help you debug issues:

**Basic Error Handling:**
```python
try:
    df = pd.read_csv('data.csv')
    print(f"Loaded {len(df)} rows of data")
except FileNotFoundError:
    print("Error: Could not find data.csv file")
    return None
```

**Simple Logging:**
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Starting data analysis")
logger.error("Failed to load data")
```

**Key Principles:**
- Use try/except blocks for operations that might fail
- Provide helpful error messages for debugging
- Log important events and errors
- Check data validity before processing
- Handle edge cases (empty data, missing files, etc.)

### Configuration
- Edit `fynesse/defaults.yml` for default configuration values
- Create `fynesse/machine.yml` for machine-specific settings
- Use `_config.yml` for project-specific configuration

### Testing
The template includes comprehensive test stubs:
```bash
# Run all tests
poetry run pytest

# Run specific module tests
poetry run pytest fynesse/tests/test_access.py

# Run with coverage
poetry run pytest --cov=fynesse
```

## Contributing

### Development Setup
1. Fork the repository
2. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
3. Install dependencies: `poetry install --with dev`
4. Create a feature branch: `git checkout -b feature/your-feature`

### Code Quality
- All code must pass tests: `poetry run pytest`
- Code must be formatted: `poetry run black fynesse/`
- Type checking must pass: `poetry run mypy fynesse/`
- Linting must pass: `poetry run flake8 fynesse/`

### Commit Guidelines
- Use conventional commit messages
- Include tests for new functionality
- Update documentation as needed

## License

MIT License - see LICENSE file for details.

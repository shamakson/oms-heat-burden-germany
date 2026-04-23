# oms-heat-burden-germany

Code and analysis for quantifying how mean-state warming increases Germany’s extreme summer heat burden using **OMS** (*Observation-based Mean-Shift*).

This repository accompanies the manuscript:

**Samakinwa, E., Scheiber, L., Cohrs, J.-C., Pfeifer, S., and Rechid, D.**  
*Mean-state warming loads Germany’s extreme summer heat burden.*

## Overview

This repository contains code to:

- process observed summer temperature data over Germany
- estimate observation-based pattern scaling against GMST
- construct counterfactual summers using OMS
- calculate extreme-heat burden (EHD)
- generate national and state-level analyses
- reproduce the manuscript figures

## Data sources

The analysis uses publicly available datasets, including:

- **E-OBS** daily gridded near-surface temperature
- **HadCRUT5** global mean surface temperature anomalies
- **Eurostat/GISCO** administrative boundaries
- **DESTATIS** population data

Please check the manuscript and data documentation for exact versions and access links.

## Method summary

OMS preserves observed day-to-day weather variability while shifting only the seasonal mean state according to an observation-based local/global warming sensitivity \(\beta\). This enables construction of counterfactual pre-industrial and global-warming-level summer realizations while holding weather sequences fixed.

## Reproducibility

To reproduce the main results:

1. prepare the required input datasets
2. run preprocessing scripts
3. estimate the GMST–Germany scaling relationship
4. generate OMS counterfactual summers
5. compute EHD metrics
6. reproduce manuscript figures and tables

## Environment

Create the analysis environment and install dependencies listed in:

- `environment.yml`

## Citation

If you use this repository, please cite the associated manuscript once published.

## Contact

Eric Samakinwa  
Climate Service Center Germany (GERICS), Helmholtz-Zentrum Hereon


# EO-Pharmaceutical-Pollutants

Repository accompanying the manuscript:

"Predicting the Degradation Kinetics of Pharmaceutical Pollutants during Electrochemical Oxidation: A Synergistic Machine Learning Framework and Mechanistic Insights"

## Contents

- Dataset (355 kinetic observations from 31 pharmaceutical compounds)
- Feature selection workflow
- Traditional machine learning models
- D-MPNN model
- KANO model
- SHAP analysis

## Data Description

Data.xlsx contains:

- Pharmaceutical information (compound name, CAS number, and SMILES)
- Experimental conditions for electrochemical oxidation
- Molecular descriptors and CDFT descriptors
- Apparent first-order degradation rate constants (log2k)
- Original train/test split used in the manuscript

  
## Requirements

Python 3.11
scikit-learn
xgboost
chemprop
RDKit

## Citation

If you use these data or codes, please cite the associated publication once available.

## Note
Minor differences in feature selection results may occur across software versions and computational environments. The final feature set used for model development and all results reported in the manuscript is provided in Data.xlsx.

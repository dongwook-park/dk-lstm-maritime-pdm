# DK-LSTM: Standards-Aligned Hybrid AI–Digital Twin Framework for Maritime Predictive Maintenance

## Paper
**Title:** A Standards-Aligned Hybrid AI–Digital Twin Framework for Robust Predictive Maintenance under Data Scarcity  

## Repository Contents

| File | Description |
|------|-------------|
| `dk_lstm_experiment_v3.1.py` | Full experimental script. Reproduces all results in the manuscript. Random seed fixed at 42. |
| `results_summary.csv` | Quantitative results for all scenarios (E1, E2, E3a, E3b): RMSE, MAE, NASA Score, Neg%. |
| `iso19848_data_contract.json` | ISO 19848-aligned reference schema mapping NASA C-MAPSS sensor channels to standard channel identifiers (Supplementary Material S1). |

## Primary Dataset (Not Included — Publicly Available)
The NASA C-MAPSS turbofan degradation dataset used in this study is publicly available:  
**Source:** https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data  
**Subsets used:** FD001, FD002, FD003

## Requirements
- Python 3.x
- TensorFlow >= 2.x
- NumPy
- pandas
- scikit-learn
- matplotlib

## Reproducibility
All experiments use fixed random seed (42) and fixed train/validation splits.  
To reproduce results, download the C-MAPSS dataset from the NASA source above,  
place the files in the working directory, and run `dk_lstm_experiment_v3.1.py`.

## Notes
- No personally identifiable or proprietary information is included.
- `iso19848_data_contract.json` is a reference mapping inspired by ISO 19848 naming principles and does not claim full technical compliance with the standard.

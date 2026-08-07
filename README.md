# Battery SoH + RUL Estimator

5-person hackathon build. MVP scope, synthetic data only.

## Structure
- `src/generate_data.py` — synthetic telemetry generator (Person A)
- `src/features.py` — feature engineering (Person B)
- `src/soh_model.py` — SoH estimation model (Person C)
- `src/rul_model.py` — RUL prediction + uncertainty bands (Person D)
- `src/narrative.py`, `app.py` — narrative layer + dashboard (Person E)
- `data/` — generated CSVs
- `notebooks/` — exploration

## Data contract
See team plan doc. Do not change column names without telling the team.

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/generate_data.py
```

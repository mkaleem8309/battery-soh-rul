# 🔋 Battery State-of-Health (SoH) & Remaining Useful Life (RUL) Estimator

A data-driven telemetry analytics platform and predictive machine learning dashboard for estimating battery capacity degradation, remaining operational life, and safety-aware operator diagnostics.

---

## 👥 Team Module Ownership
- `src/generate_data.py` — Synthetic Telemetry Generator (Person A)
- `src/features.py` — Feature Engineering (Person B)
- `src/soh_model.py` — SoH Estimation Model (Person C)
- `src/rul_model.py` — RUL Prediction & Uncertainty Bands (Person D)
- `src/narrative.py`, `app.py` — AI Safety Narrative & Streamlit Dashboard (Person E)

---

## 📌 Problem Statement
Battery energy storage systems (BESS) and electric vehicle (EV) packs suffer non-linear capacity loss due to thermal stress, deep depth-of-discharge (DoD), and high C-rate operation. Predicting State-of-Health (SoH) and Remaining Useful Life (RUL) with reliable uncertainty bounds is critical for preventing premature failures while ensuring operational safety.

---

## 🛠️ Approach & System Architecture

1. **Synthetic Telemetry Generator (`src/generate_data.py`)**:
   Simulates multi-cell per-cycle telemetry over 400 cycles with non-linear degradation physics.

2. **Feature Engineering (`src/features.py`)**:
   Extracts physical predictors including internal resistance proxy ($\Delta V / \Delta I$), cumulative thermal exposure ($T_{\text{max}} > 40^\circ\text{C}$), C-rate, and DoD without numerical instability.

3. **SoH Degradation Modeling (`src/soh_model.py`)**:
   - **Primary (Labeled)**: `RandomForestRegressor` mapping features $\rightarrow$ ground-truth SoH.
   - **Fallback (Unlabeled)**: Integration of discharge current over time, normalized to first-cycle capacity with rolling window smoothing.

4. **Driver Identification & Physics Audit**:
   Ranks degradation stressors by Random Forest feature importance and cross-checks with Pearson correlation.

5. **RUL Prediction with Uncertainty Bands (`src/rul_model.py`)**:
   Extrapolates trailing degradation slopes to the **80% EOL threshold**, computing **Worst Case (P10)**, **Likely (P50)**, and **Best Case (P90)** remaining cycles.

6. **Safety-Aware LLM Narrative (`src/narrative.py`)**:
   Integrates local Ollama (`llama3.2:3b`) with system-level guardrails to generate 2-3 sentence operator summaries. Hardened against prompt injection to strictly block any recommendation of BMS overrides or unsafe operational bypasses.

7. **Interactive Analytics Dashboard (`app.py`)**:
   Built with Streamlit and Plotly, displaying persistent safety warnings, KPI summary cards, confidence band trajectories, driver rankings, RUL uncertainty bars, and live AI operator summaries.

---

## ⚡ How to Run

1. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Ensure Ollama Local Model is Active**:
   ```powershell
   ollama run llama3.2:3b
   ```

3. **Launch Streamlit Dashboard**:
   ```powershell
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 📋 Key Assumptions & Limitations

- **Future Usage Continuity**: RUL extrapolation assumes future cell operating conditions remain consistent with trailing 50-cycle history.
- **EOL Cutoff**: Standard End-of-Life is fixed at **80.0% SoH**.
- **Synthetic Telemetry**: Built for MVP validation using simulated multi-profile telemetry.

---

## 🛡️ Safety Guardrails & Disclaimer

- **Persistent Disclaimer**:
  > *"⚠️ SAFETY DISCLAIMER: This is a data-driven estimate, not a substitute for manufacturer testing or certified diagnostics. Do not use this tool as the sole basis for safety-critical decisions."*
- **Refusal System**: System-prompt instructions strictly forbid the LLM from suggesting BMS bypasses or exceeding rated thermal/current limits, defaulting to `"Monitor Closely"` under high uncertainty.

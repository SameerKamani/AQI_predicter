## Pearls AQI Predictor – End‑to‑End Air Quality Forecasting for Karachi

### What this project is
An end‑to‑end, production‑grade system that forecasts Karachi’s Air Quality Index (AQI) for the next three days. It includes data acquisition, feature engineering, a multi‑model training and blending pipeline, automated CI/CD, an interactive web app, interpretability with SHAP, and containerized deployment.

---

### Why it matters
Air quality affects daily life. Accurate short‑term forecasts help people and organizations plan outdoor activities, mitigate exposure, and trigger early alerts when air becomes hazardous. This project demonstrates how to build such a system with modern, reliable MLOps practices.

---

## High‑level architecture

- Data ingestion and backfill from external sources into a local feature store
- Offline feature store (Parquet) and online serving with Feast
- Model training for multiple algorithms per forecast horizon (D+1, D+2, D+3)
- Blending to combine model strengths and improve accuracy
- Automated pipelines on GitHub Actions: hourly data updates, daily training
- FastAPI backend with a Gradio dashboard for real‑time predictions and visualization
- SHAP‑based global explanations for model interpretability
- Containerized for consistent deploys

---

## Data and feature engineering

- Sources
  - OpenWeather hourly air‑pollution endpoints for pollutants and category indices
  - Open‑Meteo ERA5 hourly weather archive for historical meteorology

- Processing
  - Raw hourly pollutant readings are cleaned, deduplicated, and aggregated to daily statistics (means, counts, eight‑hour O3 metric).
  - EPA‑style AQI is computed from pollutant sub‑indices and used both as a label and a predictive feature with careful time alignment.
  - Calendar features (day, month, week‑of‑year, weekday/weekend) and seasonal encodings (sine/cosine of day‑of‑year) are added.
  - Time‑aware signals include lags for AQI and particulates, rolling means and standard deviations across multiple windows, weather aggregates and limited weather lags/rollings.
  - Targets are created for the next three days at the daily grain: D+1, D+2, D+3.

- Storage
  - Offline store: Parquet and CSV under `Data/feature_store/` for full training windows and analysis.
  - Online store: Feast reads from the offline Parquet and materializes to an online database for low‑latency inference. The feature view includes all training inputs so that inference matches training columns.

---

## Modeling

- Per‑horizon training (hd1, hd2, hd3)
  - LightGBM gradient boosting with optional Optuna tuning and early stopping
  - Histogram‑based Gradient Boosting Regressor (scikit‑learn) with optional tuning
  - Linear baselines (Ridge, ElasticNet, Huber) with robust scaling and time‑series CV model selection
  - Random Forest regressors with optional Optuna tuning

- Evaluation
  - Metrics include RMSE, MAE, R², mean absolute percentage error, EPA category accuracy, and rate of predictions within 15 AQI points.
  - All metrics per horizon are summarized under `EDA/*_output/summary.json`.

- Interpretability
  - SHAP global importance is produced for LightGBM, Linear, and Random Forest, and for HGBR using a model‑agnostic SHAP explainer. Outputs live under `EDA/shap_output/`.

- Blending
  - Stacking script aligns holdout predictions by timestamp and solves a least‑squares system for optimal weights (with optional non‑negativity and simplex constraints, and optional isotonic calibration).
  - Current blend typically favors LightGBM while leveraging complementary gains from HGBR and RF. Blend metrics and per‑horizon weights are saved to `EDA/blend_output/summary.json` and `Models/registry/blend_weights_*.json`.

---

## Automation (CI/CD)

- Hourly features
  - A workflow runs the feature pipeline every hour to ingest the current day, deduplicate, and materialize to the online store. This maintains near‑real‑time features for inference.

- Daily training
  - A scheduled workflow retrains LightGBM, HGBR, Linear, and Random Forest models, writes predictions and metrics, recomputes blend weights, generates a latest forecast JSON, and uploads all artifacts.
  - An alert step marks the run if the blended D+1 forecast is hazardous (AQI ≥ 200), increasing visibility when action is needed.

- Container build and push
  - A workflow builds the application image and pushes both a short‑SHA and a latest tag to the configured container registry.

---

## Web application

- Backend
  - FastAPI REST endpoints: health, latest features, latest predictions. For inference, the service fetches the most recent feature vector from Feast’s online store and uses the latest models and blend weights from the model registry.

- Dashboard
  - Gradio UI mounted under the FastAPI server shows the blended forecasts for the next three days, a last‑30‑days line chart with forecast dots, and a red banner alert if the next‑day forecast is hazardous.

- Artifacts
  - The app is configured by environment variables pointing to the feature repo, the feature Parquet path, and the model registry. It is suitable for running locally or in a containerized environment.

---

## Artifacts and registry

- Model registry under `Models/registry/` stores trained models, per‑horizon prediction CSVs, metadata, feature importances, blend weights, and the latest combined forecast.
- EDA output folders break down metrics, feature importance, and SHAP summaries per model.

---

## How to run this project (no commands shown)

1) Prerequisites
   - Install a recent Python (version aligned with `requirements.txt`) and a working compiler stack suitable for LightGBM. Ensure internet access for data fetches.
   - Obtain an OpenWeather API key and set it in the environment under the documented variable name. Ensure container runtime is installed if you plan to run via Docker.

2) Local setup
   - Create a Python virtual environment and install project dependencies from `requirements.txt`.
   - Backfill features by executing the feature pipeline script for a historical window. This produces the Parquet file under `Data/feature_store/` and SHAP/EDA results.
   - Register and apply Feast objects from `feature_repo/`, then materialize the latest window to the online store so the web app can fetch features at inference time.

3) Train models and blend
   - Run the training scripts for LightGBM, HGBR, Linear, and Random Forest using the same holdout window. Review metrics in the EDA output folders.
   - Execute the stacking script to compute blend weights and generate the combined performance report. A latest forecast JSON is written to the model registry.

4) Serve the web app
   - Start the FastAPI server that mounts the Gradio dashboard. Open the documented UI path in the browser to view the last‑30‑days chart, 3‑day forecast dots, and any hazard alert banner.

5) Automation and alerts
   - Enable the hourly features workflow to keep features current and the daily training workflow to retrain and publish a fresh forecast. Provide the necessary repository secrets (data API key and container registry credentials if using the image build workflow). The daily workflow flags hazardous next‑day AQI predictions for rapid awareness.

6) Containerization (optional)
   - Build the provided container image and run it with appropriate environment variables and a port mapping. The image exposes the FastAPI server that hosts the API and dashboard.

---

## Design choices and notes

- Time‑based validation and careful lag/rolling construction reduce leakage and improve generalization.
- The blend is deliberately lightweight and explainable, and it stores artifacts so the same weights are available to the inference path.
- Feast keeps the feature contract consistent between training and serving; expanding the feature view ensured online rows match the model’s expectations.
- SHAP outputs provide global interpretability while keeping costs manageable by sampling when necessary.

---

## Results at a glance

- The blended model improves accuracy over individual baselines, particularly at D+2 and D+3 where single‑model forecasts tend to drift.
- Per‑horizon strengths differ: LightGBM is most consistent overall, HGBR is often strong at D+1, and Random Forest adds diversity that benefits stacking.

---

## Configuration and secrets

- Data access: an environment variable for the air‑quality API key is required to run the feature pipeline and the hourly workflow.
- Container registry: two repository secrets provide credentials for the automated image build and push workflow.

---

## Troubleshooting

- If model inference complains about feature mismatches, re‑apply Feast and re‑materialize after generating features so that online rows reflect the expanded schema.
- If the web app starts but charts are empty, ensure the Parquet exists and contains the last 30 days; run the feature pipeline and re‑materialize.
- If LightGBM fails at runtime inside a container, verify that the OpenMP runtime package is present in the base image, as included in the provided container file.

---

## Roadmap

- Add an optional deep learning forecaster for comparison and research.
- Incorporate external monitoring and alert channels (email or chat) for hazard notifications.
- Extend the dashboard with station‑level breakdowns and uncertainty bands.

---

## Acknowledgements

This project brings together open data sources, the scientific Python ecosystem, and MLOps tooling. Many thanks to the maintainers of Feast, LightGBM, scikit‑learn, SHAP, FastAPI, and Gradio for their excellent work.



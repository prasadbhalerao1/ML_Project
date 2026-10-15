# Project: Loan Default Risk Prediction Using Random Forest

Act as a machine learning engineer and full-stack developer. Build a clean, reproducible, end-to-end application that predicts borrower delinquency risk using a Random Forest classifier, with a FastAPI backend and a React frontend.

The objective is to turn a basic ML experiment into a complete, well-documented full-stack project suitable for a college project and GitHub portfolio.

**Research the dataset first. Prefer Kaggle as the data source. Keep the project simple, avoid unnecessary complexity, and document everything accurately.**

# Phase 1 — Research Kaggle Datasets First

Before implementing anything, research suitable Kaggle datasets for loan default or credit delinquency prediction.

Start with this primary candidate:

**Give Me Some Credit — Official Kaggle Competition**

[https://www.kaggle.com/competitions/GiveMeSomeCredit/data](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

This dataset uses `SeriousDlqin2yrs` as its target, indicating whether a borrower experienced serious delinquency within two years.

Investigate the dataset and verify its actual schema, target definition, data dictionary, access requirements, and terms of use before using it.

Also compare these alternatives:

1. Loan Default Prediction:
   [https://www.kaggle.com/competitions/loan-default-pred/data](https://www.kaggle.com/competitions/loan-default-pred/data)

2. Home Credit Default Risk:
   [https://www.kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Prioritize datasets according to:

- Relevance to the prediction objective.
- Availability of a clearly defined target variable.
- Dataset size and number of useful features.
- Data quality and missing-value handling requirements.
- Risk of target leakage.
- Ease of downloading and running locally.
- Dataset provenance, licensing, attribution, and access requirements.
- Suitability for a beginner-friendly Random Forest implementation.

## Dataset selection rules

Use Give Me Some Credit as the preferred starting point if its files and usage terms are accessible.

It is a reasonable fit because it provides labeled historical borrower data and avoids the need to combine numerous large tables.

However, verify the dataset rather than blindly assuming its schema. Inspect the data dictionary and actual files before selecting features.

The target means serious delinquency within two years. Do not describe the model as predicting every kind of loan default or guaranteeing whether a borrower will repay.

The official competition provides training data with labels and a separate test dataset without labels. For this educational application, create your own training, validation, and test splits from the labeled training data. Never report evaluation metrics calculated on an unlabeled competition test set.

If access requires accepting competition rules, signing into Kaggle, or configuring the Kaggle API, document the exact steps.

Do not silently replace the selected dataset with synthetic data or another dataset. If access is blocked, explain the blocker and evaluate a suitable alternative explicitly.

## Mandatory dataset documentation

Create `docs/dataset-research.md` containing:

- The selected dataset's exact title and direct Kaggle URL.
- Dataset owner or publisher, version where available, and provenance.
- Other Kaggle datasets evaluated and a short comparison table.
- Dataset dimensions and feature descriptions based on actual inspection.
- Target definition and class meanings.
- License, attribution requirements, and competition rules.
- Dataset download and setup instructions.
- Reasons for selecting the dataset.
- Known limitations, potential biases, and data-quality issues.

The same clickable Kaggle URL must appear in the root `README.md`.

Never document only that the data came from Kaggle. Include the actual link.

Complete the research first, then proceed with implementation without waiting for approval unless essential access requirements prevent progress.

# Phase 2 — Data Inspection and Preprocessing

Use:

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib and Seaborn for a few useful visualizations

Inspect the actual dataset before writing preprocessing logic.

Analyze:

- Dataset shape and column names.
- Data types.
- Missing values and missing-value percentages.
- Duplicate records.
- Target distribution and class imbalance.
- Invalid values and suspicious outliers.
- Feature distributions and correlations.
- Potential leakage between the predictors and target.

Use a few relevant charts, such as the target distribution, selected feature distributions, and a correlation heatmap. Avoid unnecessary exploratory analysis.

## Preprocessing requirements

1. Preserve the original dataset.
2. Identify and remove only verified index or identifier columns that should not be model inputs.
3. Validate the target and predictor schema.
4. Investigate missing data, including fields such as monthly income or number of dependents if they exist in the selected dataset.
5. Use an appropriate imputation strategy, such as median imputation for numeric predictors.
6. Investigate unusual values before deciding whether they need correction or treatment.
7. Do not delete records or cap outliers without a defensible reason.
8. Exclude the target and any target-derived features from the inputs.
9. Use a stratified train/validation/test split to preserve class proportions where appropriate.
10. Fit preprocessing transformations on training data only.

Use a fixed random seed so the results can be reproduced.

A simple scikit-learn Pipeline containing preprocessing and the model is preferred where practical. Avoid elaborate transformations that do not improve the project.

Document the actual cleaning decisions, discarded fields, and remaining data limitations in `docs/methodology.md`.

# Phase 3 — Train the Random Forest Model

The primary algorithm must be `RandomForestClassifier`.

Implement the training pipeline in:

`backend/ml/train.py`

## Model training requirements

1. Load the downloaded labeled dataset from a documented local path.
2. Validate the expected target and input features.
3. Separate features from the target.
4. Create training, validation, and test splits.
5. Fit preprocessing only on training data.
6. Train a reproducible Random Forest baseline.
7. Evaluate a limited number of sensible configurations if useful.
8. Investigate class weighting when appropriate for the observed class imbalance.
9. Choose the final configuration using validation results rather than repeated test-set tuning.
10. Save the fitted preprocessing and model pipeline as a reusable artifact using `joblib`.
11. Save model metadata, feature names, model version, and evaluation metrics alongside the artifact.

Do not use XGBoost, neural networks, ensembles of different algorithms, or extensive hyperparameter optimization. The purpose is to implement and understand Random Forest correctly.

## Evaluation requirements

Evaluate the final model on the held-out test set.

Report:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Precision-recall AUC, where appropriate.
- Confusion matrix.
- Class distribution in the training and test sets.

Discuss the trade-off between false positives and false negatives.

Accuracy by itself is insufficient when the positive class is relatively uncommon. Explain how class weighting or probability-threshold changes affect precision and recall.

Generate all evaluation metrics from the actual execution. Never fabricate results.

Optionally include a small feature-importance visualization, but clearly distinguish model-derived importance from causal explanations of default risk.

Create:

- `backend/ml/train.py`
- `backend/ml/evaluate.py`, if a separate evaluation script is useful.
- `backend/ml/artifacts/`
- `docs/model-evaluation.md`

# Phase 4 — Build the FastAPI Backend

Convert the trained ML pipeline into a working FastAPI application.

Use:

- FastAPI
- Pydantic
- Uvicorn
- scikit-learn
- joblib

Create these endpoints.

## `GET /health`

Return the API status, model availability, and model version.

## `GET /model-info`

Return:

- Model name and version.
- Dataset name and source URL.
- Target definition.
- Required input feature names.
- Actual evaluation metrics when available.

## `POST /predict`

Accept the exact predictors required by the trained model.

Return a response similar to:

{
"default_probability": 0.23,
"predicted_class": 0,
"risk_band": "Medium",
"model_version": "1.0",
"target_definition": "Serious delinquency within two years"
}

This is an illustrative schema, not a real prediction. All actual probabilities, labels, and model metadata must come from the trained model.

### Backend requirements

- Validate input using Pydantic.
- Match the exact feature schema used during training.
- Return useful errors for missing fields, invalid values, non-finite numbers, and unsupported inputs.
- Verify the ordering of the classes returned by the model before mapping `predict_proba` values.
- Load the model artifact once at application startup.
- Reuse the saved preprocessing pipeline during inference.
- Keep prediction logic separate from API route handlers.
- Configure CORS for the development frontend.
- Provide appropriate HTTP status codes and meaningful error messages.
- Enable FastAPI's automatic Swagger documentation.

Do not train or reload the model for every prediction request.

## Risk-band handling

Display the predicted class and estimated positive-class probability.

If adding Low, Medium, and High risk categories, implement them through documented, configurable probability thresholds. Do not present arbitrary thresholds as empirically validated lending thresholds.

Clearly distinguish:

- The predicted binary class.
- The estimated model probability.
- Any optional illustrative risk-band label.

Do not claim that model probability is a calibrated real-world probability unless calibration has been evaluated appropriately.

# Phase 5 — Build the React Frontend

Create a responsive frontend using React, Vite, and TypeScript.

Use Tailwind CSS if it simplifies consistent styling, but avoid unnecessary dependencies.

The UI should have a professional, clean design with readable typography, sensible spacing, clear forms, subtle colors, and accessible loading and error states.

## Page 1 — Dashboard

Display:

- Project overview.
- Model name.
- Dataset source.
- Target definition.
- Actual evaluation metrics when available.
- A brief explanation of what the model predicts.

## Page 2 — Default Risk Prediction

Build a user-friendly form using the actual features expected by the trained model.

Requirements:

- Convert technical feature names into understandable labels.
- Provide short explanations for less obvious financial variables.
- Group related inputs logically.
- Validate numeric fields and enforce reasonable input constraints based on the data dictionary.
- Clearly identify required fields.
- Do not add unsupported variables such as loan amount, interest rate, or employment status if they are not present in the trained feature schema.
- Keep frontend field names synchronized with the backend request schema.

Optionally include a button to load a real example record from the selected public dataset. Label it as demonstration data rather than information about the current user.

## Page 3 — Prediction Results

After a successful API request, display:

- Predicted class: serious delinquency or no serious delinquency under the dataset's target definition.
- Estimated probability of serious delinquency.
- Optional configured risk band.
- A clear, accessible probability visualization.
- A brief explanation of the result and its limitations.

Do not present a prediction as a definitive statement about a borrower's future or use it to claim that a real loan should be approved or rejected.

## Page 4 — Model Information

Show:

- Algorithm.
- Dataset source with a clickable Kaggle link.
- Target definition.
- Feature list.
- Model version.
- Actual evaluation metrics.

## Frontend engineering requirements

- Use reusable components.
- Keep API functions separate from presentation components.
- Define TypeScript interfaces for requests and responses.
- Use an environment variable for the API base URL.
- Include loading indicators, input validation, and readable API errors.
- Make the layout functional on desktop and mobile.
- Avoid fake metrics and hardcoded prediction results.

Do not implement authentication, payments, database storage, or an elaborate analytics dashboard.

# Phase 6 — Keep the Codebase Clean

Use a structure similar to:

loan-default-prediction/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── schemas/
│   │   │   └── prediction.py
│   │   ├── services/
│   │   │   └── predictor.py
│   │   └── core/
│   │       └── config.py
│   ├── ml/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── artifacts/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── .env.example
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── dataset-research.md
│   ├── methodology.md
│   ├── model-evaluation.md
│   └── limitations.md
├── .gitignore
└── README.md

Adapt this structure where useful, but retain clear separation between model training, model inference, backend APIs, and frontend components.

Do not commit raw datasets, credentials, virtual environments, `node_modules`, temporary files, or machine-specific paths. Document how to obtain the dataset and regenerate the model artifact.

Avoid duplicated code, unnecessarily tiny modules, unused dependencies, and a single giant file containing the entire application.

# Phase 7 — Documentation

The root `README.md` must include:

- Project objective and scope.
- Direct Kaggle dataset link.
- Dataset provenance and download instructions.
- License or competition-rule notes.
- Target definition and known limitations.
- Architecture and technology stack.
- Prerequisites and installation commands.
- Preprocessing methodology.
- Model training and evaluation commands.
- Commands to start the FastAPI backend and React frontend.
- API endpoints and example requests.
- Test instructions.
- Actual model evaluation results, if executed successfully.
- Limitations and possible future improvements.

Create `docs/limitations.md` discussing:

- Dataset representativeness and possible bias.
- Missing or anomalous values.
- Class imbalance.
- Potential distribution shift.
- Limitations of the model's probability estimates.
- Why the application is an educational prototype rather than a credit-underwriting system.

Documentation must reflect the actual implementation. Clearly distinguish completed work from planned improvements.

# Phase 8 — Testing and Final Verification

Before completing the project:

1. Verify the dataset source and download procedure.
2. Confirm that training uses labeled records only.
3. Run preprocessing from a clean state.
4. Verify that the target and predictors are correct.
5. Confirm there is no target leakage.
6. Train the Random Forest model and save the artifact.
7. Generate actual evaluation metrics.
8. Test model loading independently from training.
9. Start FastAPI and verify all endpoints.
10. Test valid requests and invalid inputs.
11. Start React and verify its connection to FastAPI.
12. Verify probability and class-label mappings.
13. Check frontend loading and API failure states.
14. Run the automated test suite.
15. Verify the README commands work.
16. Review the repository for broken imports, hardcoded paths, unused dependencies, and duplicated logic.

If Kaggle authentication, execution, or data access is unavailable, clearly document the exact blocker and complete whatever parts can be implemented honestly. Do not claim that training or tests succeeded when they were not executed.

# Scope Boundaries

Keep the entire project straightforward.

Do not add Docker, Kubernetes, PostgreSQL, Redis, Celery, MLflow, DVC, microservices, neural networks, or extensive hyperparameter optimization unless a specific need justifies them.

No database is required for this version. The application can load a previously trained model and return predictions directly.

Prioritize:

1. Correct Kaggle dataset selection and documentation.
2. Reliable preprocessing without data leakage.
3. Proper Random Forest training and evaluation.
4. Consistency between training and inference.
5. A working FastAPI-to-React integration.
6. Clean, maintainable code and accurate documentation.

This is an educational prototype, not a production credit decision system.

**Final deliverable:** a runnable and documented full-stack application. Finish by reporting the repository structure, actual Kaggle URL, setup commands, completed functionality, actual evaluation metrics, tests executed, and any remaining limitations.

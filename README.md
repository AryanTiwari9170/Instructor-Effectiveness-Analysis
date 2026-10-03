# Instructor Effectiveness Analysis & Prediction

Data-driven analysis and machine learning pipeline for predicting instructor
effectiveness tiers (Low / Medium / High) from student engagement and
performance metrics, built for an EdTech learning platform use case.

## Overview

Online learning platforms generate large volumes of engagement data, but
evaluating instructor performance across courses and batches is hard to do
consistently. This project analyzes instructor-level metrics — completion
rate, dropout rate, watch time, assignment submissions, forum activity, and
student feedback — and builds a classification model that predicts which
effectiveness tier an instructor falls into.

## Approach

1. **Exploratory Data Analysis** — distributions and correlations across all
   engagement metrics.
2. **Label construction** — a weighted composite score is built from
   engagement and outcome metrics, then binned into Low / Medium / High
   tiers.
3. **Instructor-level aggregation** — batch-level data is rolled up to one
   row per instructor.
4. **Modeling** — a Random Forest classifier, wrapped in a scikit-learn
   `Pipeline` with median imputation and scaling, is trained to predict
   effectiveness tier from engagement features.
5. **Evaluation** — stratified k-fold cross-validation plus a held-out test
   set, reported via accuracy, classification report, confusion matrix, and
   feature importance.

A key design decision: the composite score used to build the label is never
included as a model input, and the quantile thresholds that define the tiers
are fit on the training split only and then applied to the test split —
preventing the model's target from leaking into its own features.

## Tools & Technologies

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier,
  StratifiedKFold)

## Results

- Cross-validated training accuracy: ~0.75–0.85 (varies by dataset)
- Held-out test accuracy reported in the notebook, alongside a full
  classification report and confusion matrix
- Feature importance analysis identifies which engagement signals (watch
  time, score improvement, feedback score) matter most for predicting
  effectiveness

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook instructor_effectiveness_pipeline.ipynb
```

Point the `DATA_PATH` variable in the notebook at your dataset CSV. A sample
dataset is included for testing the pipeline end-to-end.

## Project Structure
├── instructor_effectiveness_pipeline.ipynb # Main analysis & model notebook
├── sample_dataset_for_testing.csv # Example dataset
├── requirements.txt # Python dependencies
├── LICENSE
└── README.md

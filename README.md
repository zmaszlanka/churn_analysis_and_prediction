# Churn Analysis & Prediction

## Project Overview

This repository simulates a full data science workflow for customer churn analysis at a digital bank. It covers synthetic data generation, exploratory analysis, feature engineering, customer segmentation, predictive modelling, and business recommendations. The project is designed for reproducibility and clarity, mirroring real-world DS/ML best practices.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Workflow](#workflow)
  - [1. Synthetic Data Generation](#1-synthetic-data-generation)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Feature Engineering & Preprocessing](#3-feature-engineering--preprocessing)
  - [4. Customer Segmentation](#4-customer-segmentation)
  - [5. Modelling & Evaluation](#5-modelling--evaluation)
  - [6. Business Impact & Interpretation](#6-business-impact--interpretation)
- [AI-Assisted Development Log](#ai-assisted-development-log)
- [Deliverables](#deliverables)
- [License](#license)

## Project Structure

```
churn_analysis_and_prediction/

data/                            # all csv data created during development 
├── customer_churn_synthetic.csv     
├── customer_churn_synthetic_preprocessed.csv   
├── customer_churn_synthetic_preprocessed_with_pca.csv   
├── llm_generated_customers.csv   
└── llm_generated_customers_preprocessed.csv       
documents/
└── assignement.md               # assignment for AI agent context
saved_models/                    # trqained models
├── GradientBoosting.joblib
├── LogisticRegression.joblib
├── RandomForest.joblib
└── scaler.joblib
src/
│   ├── clustering/
│   │   └── profiler_and_cluster_building.py
│   ├── modelling/
│   │   └── model_factory.py
│   ├── preprocessing/
│   │   └──  data_preprocessing.py
│   └── synthetic_dataset/
│       ├── dataset_exploration.ipynb
│       ├── dataset_generation.py
│       └──  dataset_schema_definition.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/zmaszlanka/churn_analysis_and_prediction.git
   cd churn_analysis_and_prediction
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Workflow

### 1. Synthetic Data Generation

- **Script:** `src/synthetic_dataset/dataset_generation.py`
- **Schema:** `src/synthetic_dataset/dataset_schema_definition.py`
- **Output:** `customer_churn_synthetic.csv`
- **Description:** Generates a realistic synthetic dataset with specified number of customers that covers following aspects: 
    - customer demographics, 
    - financial behaviour,
    - measured satisfaction (nps survey)
    - digital engagement. 
Churn is modelled with business logic and random noise.

### 2. Exploratory Data Analysis (EDA)

- **Notebook:** `src/synthetic_dataset/dataset_exploration.ipynb`
- **Description:** Notebook that extensivly explore the dataset providing metrics, data previews and charts.

### 3. Feature Engineering & Preprocessing

- **Script:** `src/preprocessing/data_preprocessing.py`
- **Description:** Highly modular script that performs and to end dataset preprocessing in a configurable and reproducible way:
    - Builds engineered features (ratios, buckets, interactions), 
    - Allows easy implementation of new features
    - Implements categorical variables encoding
    - Performs scaling
    - Allows data imputation
    - performs PCA
    - Builds reproducible pipelines with custom configurations

### 4. Customer Segmentation

- **Script:** `src/clustering/profiler_and_cluster_building.py`
- **Description:** Module that performs clustering and segments profiling:
    - Segments profiler: performs out of the box cross analysis between groups
    - Clusterin wrapper: enables experimentation with different clusterings (k-means, dbscan)

### 5. Modelling & Evaluation

- **Script:** `src/modelling/model_factory.py`
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting (saved in `saved_models/`)
- **Description:** Trains and evaluates models using cross-validation and metrics (ROC AUC, precision/recall, top-k recall). Compares models and allow user to select the best for deployment.


## Deliverables

- Synthetic dataset (`customer_churn_synthetic.csv`)
- Python notebook for data exploration
- Python modules for each requirement
- Trained model files (`saved_models/`)
- `requirements.txt` and this `README.md`
- PResentation as report with business ingights and AI usage report

## License

This project is for educational and demonstration purposes.




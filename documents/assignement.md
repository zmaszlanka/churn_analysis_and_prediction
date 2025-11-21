# Assignment: Churn Analysis & Prediction

## Description
You will act as a Data Scientist at a digital bank. Since no production data is available, your first task is to create a realistic synthetic customer churn dataset that simulates the bank’s customer base. After generating and validating the data, you will explore it, engineer features and customer segments, and build predictive churn models.

The system you deliver should mirror a real applied DS/ML workflow.

## Functional Requirements

1. **Synthetic Dataset Creation**
   - Create a synthetic customer dataset from scratch (minimum **10,000 customers**) representing a digital retail bank.
   - Required feature groups (you define specific fields):
     - Demographics
     - Account / product usage
     - Financial behaviour (balances, spending, transactions)
     - Support / satisfaction signals
     - A binary churn label generated via a plausible mechanism
   - Ensure:
     - Realistic distributions and correlations
     - Clear business logic behind churn behaviour
     - No target leakage
   - Save the final dataset in a clean analytical format (CSV or Parquet).
   - Document how you modelled churn and the assumptions behind your synthetic data generation.

2. **Exploratory Data Analysis**
   - Validate schema, types, distributions, missing values, and anomalies.
   - Explore relationships between features and churn.
   - Provide clear visualisations (histograms, boxplots, correlation views).
   - Summarise the main behavioural differences between churners and non-churners.

3. **Feature Engineering**
   - Build at least **5–8 engineered features** (e.g., ratios, buckets, interactions, behavioural indicators).
   - Create a reproducible preprocessing pipeline.
   - Explain the business rationale for the new features.

4. **Customer Segmentation**
   - Create rule-based segments (3–6 groups) and profile them (size, attributes, churn rate).
   - Build at least one data-driven clustering model (e.g., K-Means) on selected features.
   - Identify high-risk or high-value segments and propose retention actions.

5. **Modelling**
   - Train:
     - A baseline model (e.g., logistic regression).
     - At least two stronger models (e.g., Random Forest, Gradient Boosting).
   - Use appropriate validation and metrics (ROC AUC, precision/recall, top-k recall).
   - Compare models clearly and justify your final choice.

6. **Business Impact & Interpretation**
   - Define a targeting strategy (e.g., contacting top 15% highest-risk customers).
   - Estimate how many churners the strategy would capture.
   - Provide model interpretability (e.g., feature importances or SHAP).
   - Present actionable recommendations for customer retention.

7. **AI-Assisted Development Log**
   - Document where and how AI assisted your work with example prompts.
   - Describe where AI suggestions were incorrect or needed adjustments.
   - Include design choices and trade-offs.

## Deliverables

- Synthetic dataset with documentation of how it was generated.
- Notebooks or scripts for:
  - EDA
  - Feature engineering
  - Segmentation
  - Modelling and evaluation
- A short report (2–4 pages) with insights and recommendations.
- `requirements.txt` and clear run instructions.
- AI usage self-assessment.

## Evaluation Criteria

- Quality, realism, and coherence of the synthetic dataset.
- Depth and clarity of EDA, segmentation, and insights.
- Strength of feature engineering and modelling pipeline.
- Code organisation, reproducibility, and clarity.
- Business reasoning and quality of recommendations.
- Effective and critical use of AI tools.

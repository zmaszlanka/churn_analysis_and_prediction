import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Optional

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib


# ============================================================================================
#  CONFIG: Modify this to add/remove/adjust business features and preprocessing configuration
# ============================================================================================

@dataclass
class Config:
    """Configuration to define feature engineering logic and preprocessing pipeline components.
    """

    engineered_features: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=lambda: {
        "logins_ratio_3m_12m": lambda df: (df["mobile_app_logins_3m"] /
                                           df["mobile_app_logins_12m"].replace(0, np.nan)).fillna(0),
        "transactions_per_login": lambda df: (df["transaction_count_3m"] /
                                              df["mobile_app_logins_3m"].replace(0, np.nan)).fillna(0),
        "balance_stability_ratio": lambda df: (df["avg_monthly_balance_3m"] /
                                               df["avg_monthly_balance_12m"].replace(0, np.nan)).fillna(1),
        "credit_utilization_proxy": lambda df: (df["revolving_balance"] /
                                                df["credit_limit"].replace(0, np.nan)).fillna(0),
        "inactivity_gap": lambda df: df["days_from_last_login"] - df["days_from_last_transaction"],

        # Buckets
        "balance_bucket": lambda df: pd.qcut(df["current_balance"], q=5, labels=False, duplicates="drop"),
        "income_bucket": lambda df: pd.qcut(df["annual_income"], q=5, labels=False, duplicates="drop"),
        "spend_bucket": lambda df: pd.qcut(df["avg_monthly_spend_3m"], q=5, labels=False, duplicates="drop"),
        "activity_bucket": lambda df: pd.qcut(df["mobile_app_logins_3m"], q=5, labels=False, duplicates="drop"),

        "spend_to_income_ratio": lambda df: ((df["avg_monthly_spend_3m"] * 12) /
                                             df["annual_income"].replace(0, np.nan)).clip(0, 2).fillna(0),

        "loan_to_income_ratio": lambda df: (df["total_loan_amount"] /
                                            df["annual_income"].replace(0, np.nan)).clip(0, 5).fillna(0),

        "complaints_rate": lambda df: (df["complaints_12m"] /
                                       df["support_contacts_12m"].replace(0, np.nan)).fillna(0),

        "engagement_score": lambda df: (
            0.4 * (df["mobile_app_logins_3m"] / (df["mobile_app_logins_12m"] + 1))
            + 0.4 * (df["transaction_count_3m"] / (df["transaction_count_12m"] + 1))
            + 0.2 * (1 / (df["days_from_last_login"] + 1))
        ),
    })

    # preprocessing pipeline configuration
    target_column: str = "churn" # specify target column name
    identifier_column: str = "customer_id" # specify identifier column name
    numeric_missing_value_strategy: str = "median" #for simplicity we use the same strategy for all numeric columns
    scaler = StandardScaler() # for simplicity we use the same scaler for all numeric columns
    pca: bool = False # whether to apply PCA for dimensionality reduction
    pca_components: int = 30 # number of PCA components to keep if PCA is applied

# ============================================================
#  FEATURE ENGINEER MODULE
# ============================================================

class FeatureEngineer:
    """Modular and configurable feature generator."""

    def __init__(self, config: Config):
        self.config = config

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature_name, func in self.config.engineered_features.items():
            df[feature_name] = func(df)
        return df


# ============================================================
#  DATA PREPROCESSOR
# ============================================================

class DataPreprocessor:
    """Handles cleaning, encoding, scaling, PCA, saving, and loading."""

    def __init__(self, feature_engineer: FeatureEngineer, config: Config):
        self.feature_engineer = feature_engineer
        self.config = config
        self.pipeline: Optional[Pipeline] = None

    # ---------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning (minimal for synthetic data just to demo the capability)."""
        df = df.copy().drop_duplicates()
        return df

    # ---------------------------------------------------------

    def build_pipeline(self, df: pd.DataFrame):
        """Define preprocessing pipeline based on df column types."""
        df_features = df.drop(columns=[self.config.target_column, self.config.identifier_column])
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

        # Build transformer for numeric values
        # steps:
        # 1. impute missing values
        # 2. scale numeric features
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=self.config.numeric_missing_value_strategy)),
            ("scaler", self.config.scaler)
        ])

        # Build transformer for categorical values
        # steps:
        # 1. impute missing values
        # 2. encode categorical features
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )  
        steps = [("preprocessor", preprocessor)]

        if self.config.pca:
            steps.append(("pca", PCA(n_components=self.config.pca_components)))

        # Final pipeline
        self.pipeline = Pipeline(steps=steps)

    # ---------------------------------------------------------
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = self.clean_data(df)
        df = self.feature_engineer.add_engineered_features(df)

        id_col = df[self.config.identifier_column]
        
        X = df.drop(columns=[self.config.target_column, self.config.identifier_column])
        y = df[self.config.target_column]

        # fit + transform
        if fit:
            self.build_pipeline(df)
            X_processed = self.pipeline.fit_transform(X)
        else:
            if self.pipeline is None:
                raise RuntimeError("Pipeline not loaded or fitted. Set fit=True or load pipeline first.")
            X_processed = self.pipeline.transform(X)

        # --- FIX sparse → dense ---
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        # --- FIX: choose correct column names ---
        if self.config.pca:
            # PCA components
            col_names = [f"pc_{i+1}" for i in range(X_processed.shape[1])]
        else:
            # Feature names from preprocessing pipeline
            pre = self.pipeline.named_steps["preprocessor"]
            col_names = pre.get_feature_names_out()

        # Build final DataFrame
        processed_df = pd.DataFrame(X_processed, columns=col_names)
        processed_df[self.config.identifier_column] = id_col.values
        processed_df[self.config.target_column] = y.values

        return processed_df

    
    # we could add save/load pipeline methods here as well

# ============================================================
#  MAIN OR NOTEBOOK USAGE
# ============================================================

if __name__ == "__main__":
    # Load raw dataset
    df_raw = pd.read_csv("src\\synthetic_dataset\\generate\\llm_generated_customers.csv")

    # Init config + modules
    config = Config()
    fe = FeatureEngineer(config)
    preprocessor = DataPreprocessor(feature_engineer=fe, config=config)

    # Process data
    df_processed = preprocessor.preprocess(df_raw, fit=True)

    # Save outputs - potential improvement : dataset versioning
    df_processed.to_csv("src\\synthetic_dataset\\generate\\preprocessed.csv", index=False)
    print(df_processed.head())

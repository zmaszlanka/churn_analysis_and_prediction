from typing import Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, validator, root_validator



# Assumptions:
# - Analysis is done for one country (EUR currency)
# - Data is already aggregated per customer (fact/dimension modeling is out of scope)
# - Data is already modeled to avoid redundancy and ensure consistency
# - customer has to have at least one account
# - Bank is digital-only (app + web), no physical branches
# - Bank offers only following products: different account types, debit/credit cards, loans
# - Not all real world deppendencies are modeled to keep schema manageable

class CustomerChurnSchema(BaseModel):
    """Pydantic model representing a single digital bank customer snapshot.
    Features are all prior to churn to avoid leakage.
  
    """

    # ---------------------------
    # Customer Demographics
    # ---------------------------
    # Customer identifier: keep distribution None and provide a `formula` helper
    # so generators implementers can use the supplied formula to create IDs.
    customer_id: str = Field(
        ...,
        description="Unique customer identifier (UUID4).",
        distribution=None,
        dependent_on=None,
        formula="import uuid; uuid.uuid4().hex",
    )
    age: int = Field(..., ge=18, le=100, description="Customer age in years", distribution={"dist": "normal", "mean": 40, "sd": 12, "min": 18, "max": 100}, dependent_on=None)
    gender: Literal["male", "female", "other"] = Field(
        ...,
        description="Recorded gender category",
        distribution={
            "dist": "categorical",
            "categories": ["male", "female", "other"],
            "condition_on": "age",
            "rules": {
                "50": {"probs": {"male": 0.5, "female": 0.5, "other": 0.0}},
                "default": {"probs": {"male": 0.49, "female": 0.49, "other": 0.02}},
            },
        },
        dependent_on="age",
    )
    city_population: int = Field(..., ge=1, le=32000000, description="Population of customer's city", distribution={"dist": "normal"}, dependent_on=None)
    household_size: int = Field(..., ge=1, le=20, description="Number of people in household", distribution={"dist": "poisson", "lambda": 2}, dependent_on=None)
    annual_income: float = Field(
        ...,
        ge=0,
        description="Annual income in EUR",
        distribution={
            "dist": "normal",
            "mean": 30000,
            "sd": 5000,
            "condition_on": "education_level",
            "rules": {
                "no_formal_education": {"mean": 20000, "sd": 5000},
                "primary_education": {"mean": 25000, "sd": 6000},
                "secondary_education": {"mean": 30000, "sd": 7000},
                "bachelor": {"mean": 45000, "sd": 10000},
                "master": {"mean": 60000, "sd": 15000},
                "doctorate": {"mean": 80000, "sd": 20000},
                "unknown": {"mean": 30000, "sd": 10000},
            },
        },
        dependent_on="education_level",
    )
    education_level: Literal[
        "no_formal_education",
        "primary_education",
        "secondary_education",
        "bachelor",
        "master",
        "doctorate",
        "unknown",
    ] = Field(
        ...,
        description="Highest reported education level or unknown",
        distribution={
            "dist": "categorical",
            "condition_on": "age",
            "categories": [
                "no_formal_education",
                "primary_education",
                "secondary_education",
                "bachelor",
                "master",
                "doctorate",
                "unknown",
            ],
            "rules": {
                "18-22": {
                    "probs": {
                        "no_formal_education": 0.01,
                        "primary_education": 0.02,
                        "secondary_education": 0.20,
                        "bachelor": 0.60,
                        "master": 0.12,
                        "doctorate": 0.00,
                        "unknown": 0.05,
                    }
                },
                "23-29": {
                    "probs": {
                        "no_formal_education": 0.005,
                        "primary_education": 0.01,
                        "secondary_education": 0.15,
                        "bachelor": 0.60,
                        "master": 0.20,
                        "doctorate": 0.005,
                        "unknown": 0.03,
                    }
                },
                "30-45": {
                    "probs": {
                        "no_formal_education": 0.01,
                        "primary_education": 0.05,
                        "secondary_education": 0.25,
                        "bachelor": 0.40,
                        "master": 0.18,
                        "doctorate": 0.01,
                        "unknown": 0.10,
                    }
                },
                "46-60": {
                    "probs": {
                        "no_formal_education": 0.02,
                        "primary_education": 0.08,
                        "secondary_education": 0.35,
                        "bachelor": 0.30,
                        "master": 0.10,
                        "doctorate": 0.01,
                        "unknown": 0.14,
                    }
                },
                "61-75": {
                    "probs": {
                        "no_formal_education": 0.05,
                        "primary_education": 0.12,
                        "secondary_education": 0.40,
                        "bachelor": 0.18,
                        "master": 0.05,
                        "doctorate": 0.00,
                        "unknown": 0.20,
                    }
                },
                "76-100": {
                    "probs": {
                        "no_formal_education": 0.08,
                        "primary_education": 0.18,
                        "secondary_education": 0.45,
                        "bachelor": 0.10,
                        "master": 0.02,
                        "doctorate": 0.00,
                        "unknown": 0.17,
                    }
                },
                "default": {
                    "probs": {
                        "no_formal_education": 0.02,
                        "primary_education": 0.05,
                        "secondary_education": 0.30,
                        "bachelor": 0.40,
                        "master": 0.15,
                        "doctorate": 0.01,
                        "unknown": 0.07,
                    }
                },
            },
        },
        dependent_on="age",
    )
    employment_status: Literal[
        "employed",
        "unemployed",
        "student",
        "retired",
        "self-employed",
        "other",
    ] = Field(
        ...,
        description="Employment status category",
        distribution={
            "dist": "categorical",
            "condition_on": "age",
            "categories": [
                "employed",
                "unemployed",
                "student",
                "retired",
                "self-employed",
                "other",
            ],
            "rules": {
                "18-22": {
                    "probs": {
                        "employed": 0.20,
                        "unemployed": 0.10,
                        "student": 0.60,
                        "retired": 0.00,
                        "self-employed": 0.05,
                        "other": 0.05,
                    }
                },
                "23-29": {
                    "probs": {
                        "employed": 0.60,
                        "unemployed": 0.10,
                        "student": 0.05,
                        "retired": 0.00,
                        "self-employed": 0.20,
                        "other": 0.05,
                    }
                },
                "30-45": {
                    "probs": {
                        "employed": 0.75,
                        "unemployed": 0.05,
                        "student": 0.00,
                        "retired": 0.00,
                        "self-employed": 0.15,
                        "other": 0.05,
                    }
                },
                "46-60": {
                    "probs": {
                        "employed": 0.70,
                        "unemployed": 0.06,
                        "student": 0.00,
                        "retired": 0.02,
                        "self-employed": 0.18,
                        "other": 0.04,
                    }
                },
                "61-75": {
                    "probs": {
                        "employed": 0.30,
                        "unemployed": 0.05,
                        "student": 0.00,
                        "retired": 0.50,
                        "self-employed": 0.08,
                        "other": 0.07,
                    }
                },
                "76-100": {
                    "probs": {
                        "employed": 0.10,
                        "unemployed": 0.05,
                        "student": 0.00,
                        "retired": 0.80,
                        "self-employed": 0.02,
                        "other": 0.03,
                    }
                },
                "default": {
                    "probs": {
                        "employed": 0.60,
                        "unemployed": 0.10,
                        "student": 0.05,
                        "retired": 0.05,
                        "self-employed": 0.15,
                        "other": 0.05,
                    }
                },
            },
        },
        dependent_on="age",
    )

    # ---------------------------
    # Account / Product Usage
    # ---------------------------
    tenure_months: int = Field(..., ge=0, description="Months since account opened", distribution={"dist": "exponential", "scale": 24}, dependent_on="signup_date")
    main_account_type: Literal["standard", "savings", "student", "business"] = Field(..., distribution={"dist": "categorical"}, dependent_on=None)

    number_of_accounts: int = Field(..., ge=1, distribution={"dist": "poisson", "lambda": 1.5}, dependent_on=None)
    num_standard_accounts: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 1}, dependent_on="number_of_accounts")
    num_savings_accounts: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.5}, dependent_on="number_of_accounts")
    num_student_accounts: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.1}, dependent_on="number_of_accounts")
    num_business_accounts: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.2}, dependent_on="number_of_accounts")

    number_of_debit_cards: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 1}, dependent_on="number_of_accounts")
    number_of_credit_cards: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.5}, dependent_on="number_of_accounts")

    total_loan_amount: float = Field(0.0, ge=0, description="Outstanding principal in EUR", distribution={"dist": "lognormal"}, dependent_on="number_of_loans")
    number_of_loans: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.2}, dependent_on=None)

    # ---------------------------
    # Financial Behavior
    # ---------------------------
    current_balance: float = Field(..., description="Total balance across accounts in EUR", distribution={"dist": "lognormal"}, dependent_on=None)
    days_from_last_transaction: int = Field(None, ge=0, distribution={"dist": "exponential", "scale": 30}, dependent_on=None)

    avg_monthly_balance_3m: float = Field(..., ge=0, distribution={"dist": "lognormal"}, dependent_on="current_balance")
    avg_monthly_balance_12m: float = Field(..., ge=0, distribution={"dist": "lognormal"}, dependent_on="current_balance")

    avg_monthly_spend_3m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "lognormal",
            "condition_on": "annual_income",
            "rules": {
                "low": {"scale_factor": 0.02},
                "medium": {"scale_factor": 0.05},
                "high": {"scale_factor": 0.1},
            },
        },
        dependent_on="annual_income",
    )
    avg_monthly_spend_12m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "lognormal",
            "condition_on": "avg_monthly_spend_3m",
            "rules": {"zero": {"multiplier": 1.0}, "positive": {"multiplier": 4.0}},
        },
        dependent_on="avg_monthly_spend_3m",
    )

    transaction_count_3m: int = Field(..., ge=0, distribution={"dist": "poisson", "lambda": 5}, dependent_on=None)
    transaction_count_12m: int = Field(
        ...,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 20,
            "condition_on": "transaction_count_3m",
            "rules": {"zero": {"lambda": 0}, "positive": {"factor": 4}},
        },
        dependent_on="transaction_count_3m",
    )

    transaction_volume_3m: float = Field(..., ge=0, distribution={"dist": "lognormal"}, dependent_on="transaction_count_3m")
    transaction_volume_12m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "lognormal",
            "condition_on": "transaction_volume_3m",
            "rules": {"zero": {"multiplier": 1.0}, "positive": {"multiplier": 4.0}},
        },
        dependent_on="transaction_volume_3m",
    )

    declined_transactions_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.1}, dependent_on="transaction_count_3m")
    declined_transactions_12m: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.3,
            "condition_on": "transaction_count_12m",
            "rules": {"zero": {"lambda": 0}, "positive": {"factor": 0.01}},
        },
        dependent_on="transaction_count_12m",
    )

    credit_limit: Optional[float] = Field(None, ge=0, distribution=None, dependent_on=None)
    revolving_balance: Optional[float] = Field(None, ge=0, distribution=None, dependent_on="credit_limit")
    credit_utilization_ratio: Optional[float] = Field(None, ge=0, le=1, distribution=None, dependent_on="revolving_balance")

    loan_payments_3m: float = Field(0.0, ge=0, distribution={"dist": "lognormal"}, dependent_on="loan_payments_12m")
    loan_payments_12m: float = Field(
        0.0,
        ge=0,
        distribution={
            "dist": "lognormal",
            "condition_on": "total_loan_amount",
            "rules": {"zero": {"multiplier": 0.0}, "positive": {"fraction_of_total": 0.1}},
        },
        dependent_on="total_loan_amount",
    )

    # ---------------------------
    # Customer Support & Satisfaction
    # ---------------------------

    complaints_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.01}, dependent_on="support_contacts_3m")
    complaints_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.02}, dependent_on="support_contacts_12m")

    support_contacts_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.2}, dependent_on="mobile_app_logins_3m")
    support_contacts_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 0.5}, dependent_on="mobile_app_logins_12m")

    last_support_contact_days: Optional[int] = Field(
        None,
        ge=0,
        distribution={
            "dist": "exponential",
            "scale": 30,
            "condition_on": "support_contacts_3m",
            "rules": {"zero": {"scale": 365}, "positive": {"scale": 30}},
        },
        dependent_on="support_contacts_3m",
    )

    received_satisfaction_survey: bool = Field(False, distribution={"dist": "bernoulli", "p": 0.1}, dependent_on=None)
    filled_satisfaction_survey: bool = Field(False, distribution={"dist": "bernoulli", "p": 0.05}, dependent_on="received_satisfaction_survey")

    nps_segment: Optional[Literal["promoter", "passive", "detractor"]] = Field(None, distribution={"dist": "categorical"}, dependent_on="avg_satisfaction_score")

    # ---------------------------
    # Digital Engagement
    # ---------------------------
    days_from_last_login: int = Field(
        None,
        ge=0,
        distribution={"dist": "exponential", "scale": 30},
        dependent_on=["days_from_last_app_login", "days_from_last_web_login"],
    )
    days_from_last_web_login: Optional[int] = Field(
        None,
        ge=0,
        distribution={
            "dist": "exponential",
            "scale": 30,
            "condition_on": "online_banking_logins_3m",
            "rules": {"zero": {"scale": 365}, "positive": {"scale": 30}},
        },
        dependent_on="online_banking_logins_3m",
    )
    days_from_last_app_login: Optional[int] = Field(
        None,
        ge=0,
        distribution={
            "dist": "exponential",
            "scale": 30,
            "condition_on": "mobile_app_logins_3m",
            "rules": {"zero": {"scale": 365}, "positive": {"scale": 30}},
        },
        dependent_on="mobile_app_logins_3m",
    )

    mobile_app_logins_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 3}, dependent_on=None)
    mobile_app_logins_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 30}, dependent_on=None)

    online_banking_logins_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 2}, dependent_on=None)
    online_banking_logins_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 25}, dependent_on=None)

    avg_time_spent_per_login_minutes_3m: float = Field(0.0, ge=0, distribution={"dist": "normal", "mean": 5, "sd": 2}, dependent_on=None)
    avg_time_spent_per_login_minutes_12m: float = Field(0.0, ge=0, distribution={"dist": "normal", "mean": 6, "sd": 3}, dependent_on=None)

    push_clicks_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 1}, dependent_on=None)
    push_clicks_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 5}, dependent_on=None)

    marketing_emails_opened_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 2}, dependent_on=None)
    marketing_emails_opened_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 20}, dependent_on=None)

    # ---------------------------
    # Target Variable
    # ---------------------------
    churn: bool = Field(False, description="Business-defined churn label", distribution={"dist": "bernoulli", "p": 0.1}, dependent_on=["days_from_last_activity","days_from_last_login"])



    #TODO: think, update and check validators
    
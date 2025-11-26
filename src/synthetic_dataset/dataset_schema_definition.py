from typing import Optional, Literal
from pydantic import BaseModel, Field


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
 
    customer_id: str = Field(
        ...,
        description="Unique customer identifier (UUID4).",
        distribution=None,
        dependent_on=None,
        formula="uuid.uuid4().hex",
    )

    age: int = Field(
        ...,
        description="Customer age in years",
        type=int,
        distribution={"dist": "normal", "mean": 40, "sd": 12, "min": 18, "max": 100},
        dependent_on=None,
    )

    gender: Literal["male", "female"] = Field(
        ...,
        description="Recorded gender category",
        type=str,
        distribution={
            "dist": "categorical",
            "categories": ["male", "female"],
            "probs": {"male": 0.5, "female": 0.5},
        },
        dependent_on=None,
    )

    city_population: int = Field(
        ...,
        description="Population of customer's city",
        type=int,
        distribution={
            "dist": "normal",
            "min": 500,
            "mean": 200000,
            "sd": 100000,
            "max": 32000000,
        },
        dependent_on=None,
    )

    household_size: int = Field(
        ...,
        description="Number of people in household",
        type=int,
        distribution={"dist": "poisson", "lambda": 2, "min": 1, "max": 7},
        dependent_on=None,  # could be dependent on city size but skipped for simplicity
    )

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
        type=str,
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
                "x >= 18 and x < 22": {
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
                "x >= 23 and x < 29": {
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
                "x >= 30 and x <= 45": {
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
                "x >= 46 and x <= 60": {
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
                "x >= 61 and x <= 75": {
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
                "x >= 76 and x <= 100": {
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
        # could be dependent on education but skipped for simplicity
        description="Employment status category",
        type=str,
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
                "x >= 18 and x < 22": {
                    "probs": {
                        "employed": 0.20,
                        "unemployed": 0.10,
                        "student": 0.60,
                        "retired": 0.00,
                        "self-employed": 0.05,
                        "other": 0.05,
                    }
                },
                "x >= 23 and x < 29": {
                    "probs": {
                        "employed": 0.60,
                        "unemployed": 0.10,
                        "student": 0.05,
                        "retired": 0.00,
                        "self-employed": 0.20,
                        "other": 0.05,
                    }
                },
                "x >= 30 and x <= 45": {
                    "probs": {
                        "employed": 0.75,
                        "unemployed": 0.05,
                        "student": 0.00,
                        "retired": 0.00,
                        "self-employed": 0.15,
                        "other": 0.05,
                    }
                },
                "x >= 46 and x <= 60": {
                    "probs": {
                        "employed": 0.70,
                        "unemployed": 0.06,
                        "student": 0.00,
                        "retired": 0.02,
                        "self-employed": 0.18,
                        "other": 0.04,
                    }
                },
                "x >= 61 and x <= 75": {
                    "probs": {
                        "employed": 0.30,
                        "unemployed": 0.05,
                        "student": 0.00,
                        "retired": 0.50,
                        "self-employed": 0.08,
                        "other": 0.07,
                    }
                },
                "x >= 76 and x <= 100": {
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

    tenure_months: int = Field(
        ...,
        ge=0,
        description="Months since account opened",
        distribution={"dist": "exponential", "scale": 24},
        dependent_on=None,
    )

    main_account_type: Literal["standard", "savings", "student", "business"] = Field(
        ...,
        distribution={
            "dist": "categorical",
            "categories": ["standard", "savings", "student", "business"],
            "rules": {
                "x == 'no_formal_education'": {
                    "probs": {"standard": 0.7, "savings": 0.2, "student": 0.0, "business": 0.1}
                },
                "x == 'primary_education'": {
                    "probs": {"standard": 0.9, "savings": 0.04, "student": 0.0, "business": 0.06}
                },
                "x == 'secondary_education'": {
                    "probs": {"standard": 0.8, "savings": 0.03, "student": 0.12, "business": 0.05}
                },
                "x == 'bachelor'": {
                    "probs": {"standard": 0.8, "savings": 0.1, "student": 0.55, "business": 0.05}
                },
                "x == 'master'": {
                    "probs": {"standard": 0.8, "savings": 0.12, "student": 0.0, "business": 0.08}
                },
                "x == 'doctorate'": {
                    "probs": {"standard": 0.8, "savings": 0.1, "student": 0.0, "business": 0.1}
                },
                "default": {
                    "probs": {"standard": 0.8, "savings": 0.05, "student": 0.1, "business": 0.05}
                },
            },
        },
        dependent_on="employment_status",
    )

    num_standard_accounts: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 1,
            "min": 0,
            "condition_on": "main_account_type",
            "rules": {"standard": {"lambda": 2}, "savings": {"lambda": 1}, "student": {"lambda": 0}, "business": {"lambda": 1}, "default": {"lambda": 1}},
        },
        dependent_on="main_account_type",
    )

    num_savings_accounts: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.5,
            "min": 0,
            "condition_on": "main_account_type",
            "rules": {"standard": {"lambda": 1}, "savings": {"lambda": 2}, "student": {"lambda": 0}, "business": {"lambda": 0}, "default": {"lambda": 0.5}},
        },
        dependent_on="main_account_type",
    )

    num_student_accounts: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.1,
            "min": 0,
            "condition_on": "main_account_type",
            "rules": {"standard": {"lambda": 0}, "savings": {"lambda": 0}, "student": {"lambda": 1}, "business": {"lambda": 0}, "default": {"lambda": 0.1}},
        },
        dependent_on="main_account_type",
    )

    num_business_accounts: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.2,
            "min": 0,
            "condition_on": "main_account_type",
            "rules": {"standard": {"lambda": 0}, "savings": {"lambda": 0}, "student": {"lambda": 0}, "business": {"lambda": 2}, "default": {"lambda": 0.2}},
        },
        dependent_on="main_account_type",
    )

    num_of_accounts: int = Field(
        1,
        description="Total number of accounts held",
        distribution=None,
        dependent_on=["num_standard_accounts", "num_savings_accounts", "num_student_accounts", "num_business_accounts"],
        formula=(
            "row['num_standard_accounts'] + row['num_savings_accounts'] + "
            "row['num_student_accounts'] + row['num_business_accounts']"
        ),
    )

    number_of_debit_cards: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 1,
            "min": 0,
            "condition_on": "num_of_accounts",
            "rules": {
                "x<2": {"lambda": 1},
                "x>=2 and x<4": {"lambda": 3},
                "x>=4 and x<7": {"lambda": 4},
                "x>=7 and x<11": {"lambda": 5},
                "default": {"lambda": 1},
            },
        },
        dependent_on="num_of_accounts",
    )

    number_of_credit_cards: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.5,
            "min": 0,
            "condition_on": "num_of_accounts",
            "rules": {
                "x<2": {"lambda": 0},
                "x>=2 and x<4": {"lambda": 0},
                "x>=4 and x<7": {"lambda": 1},
                "x>=7 and x<11": {"lambda": 2},
                "default": {"lambda": 0.5},
            },
        },
        dependent_on="num_of_accounts",
    )

    total_loan_amount: float = Field(
        0,
        description="Outstanding principal in EUR",
        distribution={
            "dist": "normal",
            "sd": 10000,
            "mean": 0,
            "min": 0,
            "rules": {
                "x==0": {"sd": 0, "mean": 0, "min": 0},
                "x>=1 and x<=2": {"sd": 100000, "mean": 200000, "min": 0},
                "x>=3": {"sd": 100000, "mean": 1000000, "min": 0},
                "default": {"sd": 20000, "mean": 200000, "min": 0},
            },
        },
        dependent_on="number_of_loans",
    )

    number_of_loans: int = Field(
        0,
        distribution={"dist": "poisson", "lambda": 0.2, "min": 0},
        dependent_on=None,
    )

    # ---------------------------
    # Financial Behavior
    # ---------------------------

    current_balance: float = Field(
        ...,
        description="Total balance across accounts in EUR",
        distribution={
            "dist": "normal",
            "mean": 1000,
            "sd": 500,
            "min": 0,
            "max": 10000000000000,
        },
        dependent_on=None,
    )

    days_from_last_transaction: int = Field(
        None,
        ge=0,
        distribution={"dist": "exponential", "scale": 30, "min": 0},
        dependent_on=None,
    )

    avg_monthly_balance_3m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "current_balance",
            "rules": {
                "x<500": {"mean": 200, "sd": 50, "min": 0, "max": 10000000000000},
                "x>=500 and x<2000": {"mean": 1000, "sd": 300, "min": 0, "max": 10000000000000},
                "x>=2000": {"mean": 2500, "sd": 500, "min": 0, "max": 10000000000000},
                "default": {"mean": 1000, "sd": 300, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="current_balance",
    )

    avg_monthly_balance_12m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "current_balance",
            "rules": {
                "x<500": {"mean": 180, "sd": 50, "min": 0, "max": 10000000000000},
                "x>=500 and x<2000": {"mean": 950, "sd": 250, "min": 0, "max": 10000000000000},
                "x>=2000": {"mean": 2400, "sd": 400, "min": 0, "max": 10000000000000},
                "default": {"mean": 1000, "sd": 300, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="current_balance",
    )

    avg_monthly_spend_3m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "annual_income",
            "rules": {
                "x<30000": {"mean": 500, "sd": 100, "min": 0, "max": 10000000000000},
                "x>=30000 and x<60000": {"mean": 1500, "sd": 300, "min": 0, "max": 10000000000000},
                "x>=60000": {"mean": 4000, "sd": 800, "min": 0, "max": 10000000000000},
                "default": {"mean": 1500, "sd": 300, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="annual_income",
    )

    avg_monthly_spend_12m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "avg_monthly_spend_3m",
            "rules": {
                "x==0": {"mean": 0, "sd": 0},
                "x>0 and x<=1000": {"mean": 2000, "sd": 200, "min": 0, "max": 10000000000000},
                "x>1000": {"mean": 5000, "sd": 500, "min": 0, "max": 10000000000000},
                "default": {"mean": 2000, "sd": 200, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="avg_monthly_spend_3m",
    )

    transaction_count_3m: int = Field(
        ...,
        ge=0,
        distribution={"dist": "poisson", "lambda": 5, "min": 0},
        dependent_on=None,
    )

    transaction_count_12m: int = Field(
        ...,
        ge=0,
        distribution={
            "dist": "poisson",
            "condition_on": "transaction_count_3m",
            "rules": {
                "x==0": {"lambda": 0, "min": 0},
                "x>0": {"lambda": 20, "min": 0},
                "default": {"lambda": 20, "min": 0},
            },
        },
        dependent_on="transaction_count_3m",
    )

    transaction_volume_3m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "transaction_count_3m",
            "rules": {
                "x==0": {"mean": 0, "sd": 0, "min": 0, "max": 10000000000000},
                "x>0 and x<=5": {"mean": 500, "sd": 100, "min": 0, "max": 10000000000000},
                "x>5": {"mean": 1500, "sd": 400, "min": 0, "max": 10000000000000},
                "default": {"mean": 1000, "sd": 300, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="transaction_count_3m",
    )

    transaction_volume_12m: float = Field(
        ...,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "transaction_volume_3m",
            "rules": {
                "x==0": {"mean": 0, "sd": 0, "min": 0, "max": 10000000000000},
                "x>0 and x<=10": {"mean": 2000, "sd": 500, "min": 0, "max": 10000000000000},
                "x>10 and x<=20": {"mean": 5000, "sd": 1000, "min": 0, "max": 10000000000000},
                "x>20": {"mean": 7000, "sd": 1500, "min": 0, "max": 10000000000000},
                "default": {"mean": 2000, "sd": 1000, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="transaction_count_12m",
    )

    declined_transactions_3m: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.1,
            "condition_on": "transaction_count_3m",
            "rules": {"x==0": {"lambda": 0, "min": 0}, "x>0": {"lambda": 0.1, "min": 0}, "default": {"lambda": 0.1, "min": 0}},
        },
        dependent_on="transaction_count_3m",
    )

    declined_transactions_12m: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "condition_on": "transaction_count_12m",
            "rules": {"x==0": {"lambda": 0, "min": 0}, "x>0": {"lambda": 0.5, "min": 0}, "default": {"lambda": 0.5, "min": 0}},
        },
        dependent_on="transaction_count_12m",
    )

    loan_payments_12m: float = Field(
        0.0,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "total_loan_amount",
            "rules": {
                "x==0": {"mean": 0, "sd": 0, "min": 0, "max": 10000000000000},
                "x>0": {"mean": 800, "sd": 800, "min": 0, "max": 10000000000000},
                "default": {"mean": 0, "sd": 0, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="total_loan_amount",
    )

    loan_payments_3m: float = Field(
        0.0,
        ge=0,
        distribution={
            "dist": "normal",
            "condition_on": "loan_payments_12m",
            "rules": {
                "x==0": {"mean": 0, "sd": 0, "min": 0, "max": 10000000000000},
                "x>0": {"mean": 200, "sd": 200, "min": 0, "max": 10000000000000},
                "default": {"mean": 0, "sd": 0, "min": 0, "max": 10000000000000},
            },
        },
        dependent_on="loan_payments_12m",
    )

    complaints_3m: int = Field(
        0,
        distribution={
            "dist": "poisson",
            "lambda": 0.01,
            "condition_on": "support_contacts_3m",
            "rules": {"x==0": {"lambda": 0}, "x==1": {"lambda": 0.01}, "x>=2": {"lambda": 0.02}, "default": {"lambda": 0.03}},
        },
        dependent_on="support_contacts_3m",
    )

    complaints_12m: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.02,
            "condition_on": "support_contacts_12m",
            "rules": {"x==0": {"lambda": 0}, "x==1": {"lambda": 0.02}, "x>=2": {"lambda": 0.03}, "default": {"lambda": 0.05}},
        },
        dependent_on="support_contacts_12m",
    )

    support_contacts_3m: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.2,
            "condition_on": "mobile_app_logins_3m",
            "rules": {"x==0": {"lambda": 0}, "x>=1 and x <5": {"lambda": 1}, "x>=5": {"lambda": 2}, "default": {"lambda": 3}},
        },
        dependent_on="mobile_app_logins_3m",
    )

    support_contacts_12m: int = Field(
        0,
        ge=0,
        distribution={
            "dist": "poisson",
            "lambda": 0.5,
            "condition_on": "mobile_app_logins_12m",
            "rules": {"x==0": {"lambda": 0}, "x>=1 and x <5": {"lambda": 1}, "x>=5": {"lambda": 2}, "default": {"lambda": 4}},
        },
        dependent_on="mobile_app_logins_12m",
    )

    last_support_contact_days: Optional[int] = Field(
        None,
        ge=0,
        distribution={
            "dist": "exponential",
            "condition_on": "support_contacts_3m",
            "rules": {"x==0": {"scale": 365}, "x>0": {"scale": 30}},
        },
        dependent_on="support_contacts_3m",
    )

    # -------------------------
    # Measured satisfaction 
    # -------------------------

    received_satisfaction_survey: bool = Field(False, distribution={"dist": "bernoulli", "p": 0.1}, dependent_on=None)

    filled_satisfaction_survey: bool = Field(
        False,
        distribution={
            "dist": "bernoulli",
            "condition_on": "received_satisfaction_survey",
            "rules": {"x": {"p": 0.5}, "not x": {"p": 0.0}, "default": {"p": 0.05}},
        },
        dependent_on="received_satisfaction_survey",
    )

    nps_segment: Optional[Literal["promoter", "passive", "detractor"]] = Field(
        None,
        distribution={
            "dist": "categorical",
            "condition_on": "avg_satisfaction_score",
            "rules": {
                "x>=9": {"probs": {"promoter": 0.8, "passive": 0.15, "detractor": 0.05}},
                "x>=7 and x <9": {"probs": {"promoter": 0.3, "passive": 0.5, "detractor": 0.2}},
                "x<7": {"probs": {"promoter": 0.1, "passive": 0.2, "detractor": 0.7}},
                "default": {"probs": {"promoter": 0.33, "passive": 0.33, "detractor": 0.34}},
            },
        },
        dependent_on="avg_satisfaction_score",
    )

    # ---------------------------
    # Digital Engagement
    # ---------------------------

    days_from_last_login: int = Field(
        None,
        ge=0,
        distribution=None, 
        dependent_on=["days_from_last_app_login", "days_from_last_web_login"],
        formula="min(row['days_from_last_app_login'], row['days_from_last_web_login'])",
    )

    days_from_last_web_login: Optional[int] = Field(
        None,
        ge=0,
        distribution={
            "dist": "exponential",
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
            "condition_on": "mobile_app_logins_3m",
            "rules": {"zero": {"scale": 365}, "positive": {"scale": 30}},
        },
        dependent_on="mobile_app_logins_3m",
    )

    mobile_app_logins_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 3, "min": 0}, dependent_on=None)

    mobile_app_logins_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 30, "min": 0}, dependent_on=None)

    online_banking_logins_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 2, "min": 0}, dependent_on=None)

    online_banking_logins_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 25, "min": 0}, dependent_on=None)

    # Time spent per login
    avg_time_spent_per_login_minutes_3m: float = Field(
        0.0, ge=0, distribution={"dist": "normal", "mean": 5, "sd": 2, "min": 0}, dependent_on=None
    )

    avg_time_spent_per_login_minutes_12m: float = Field(
        0.0, ge=0, distribution={"dist": "normal", "mean": 6, "sd": 3, "min": 0}, dependent_on=None
    )

    # Push notifications
    push_clicks_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 1, "min": 0}, dependent_on=None)
    push_clicks_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 5, "min": 0}, dependent_on=None)

    # Marketing emails
    marketing_emails_opened_3m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 2, "min": 0}, dependent_on=None)
    marketing_emails_opened_12m: int = Field(0, ge=0, distribution={"dist": "poisson", "lambda": 20, "min": 0}, dependent_on=None)

    # ---------------------------
    # Target Variable
    # ---------------------------

    # Churn risk is calculated as a weighted combination of key behavioral and financial signals.
    # Higher churn probability is assigned when customers show inactivity, low balances, low engagement,
    # or frustration signals. Each component adds or subtracts from a base churn probability:

    # • Days since last login: The longer the inactivity, the higher the churn risk.
    # • Low recent balance: Customers with very low balances are more likely to stop using the bank.
    # • Low transaction activity: No or few transactions indicate declining engagement.
    # • Low mobile app usage: Core engagement metric for a digital bank — low logins signal churn.
    # • High support contacts or complaints: Indicates dissatisfaction and increases churn risk.
    # • Low income + loan repayments: Financial stress can lead to leaving the bank.
    # • Small random noise: Adds realistic variability.
    # • Constant baseline: Ensures minimal churn probability for all customers.

    # The final risk value is clipped to [0,1] and converted into a churn=True/False using a random draw.


    churn: bool = Field(
        False,
        description="Business-defined churn label (calculated via business rules).",
        distribution=None,
        dependent_on=[
            "days_from_last_activity",
            "days_from_last_login",
            "avg_monthly_balance_3m",
            "support_contacts_12m",
            "complaints_12m",
            "transaction_count_3m",
            "mobile_app_logins_3m",
            "loan_payments_12m",
            "annual_income",
        ],
        formula="(lambda row: \
    (lambda risk: \
        (random.random() < max(0, min(1, risk))) \
    )( \
        0 \
        + (0.40 if row['days_from_last_login'] > 120 else \
           0.25 if row['days_from_last_login'] > 60 else \
           0.10 if row['days_from_last_login'] > 30 else 0) \
        + (0.15 if row['avg_monthly_balance_3m'] < 100 else \
           0.05 if row['avg_monthly_balance_3m'] < 500 else 0) \
        + (0.20 if row['transaction_count_3m'] == 0 else \
           0.05 if row['transaction_count_3m'] < 5 else 0) \
        + (0.20 if row['mobile_app_logins_3m'] == 0 else \
           0.10 if row['mobile_app_logins_3m'] < 3 else 0) \
        + (0.10 if row['support_contacts_12m'] >= 5 else 0) \
        + (0.20 if row['complaints_12m'] >= 2 else \
           0.10 if row['complaints_12m'] == 1 else 0) \
        + (0.10 if (row['loan_payments_12m'] > 0 and row['annual_income'] < 30000) else 0) \
        + (random.uniform(-0.05, 0.05)) \
        + 0.05 \
    ) \
    )(row)"
    )

    # possilbe alternative approach: classify subsamle as churn (manually or with GenAI) and then train a model to predict it

    # possible enhancements to pydantic model: add logical validators

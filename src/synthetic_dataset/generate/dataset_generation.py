import pandas as pd
import numpy as np
import uuid
from typing import Any
from pydantic import BaseModel
from dataset_schema_definition import CustomerChurnSchema
import re
import random


# ============================================================
# 1. Dependency resolution
# ============================================================

class DependencyResolver:
    def __init__(self, schema_cls: BaseModel):
        self.schema_cls = schema_cls
        self.fields = schema_cls.model_fields
        self.graph = {}
        self.order = []

    def build_graph(self):
        """Extract dependency mapping from json_schema_extra"""
        for name, field in self.fields.items():
            extra = field.json_schema_extra or {}
            dep = extra.get("dependent_on")
            if dep is None:
                self.graph[name] = []
            elif isinstance(dep, list):
                self.graph[name] = dep
            else:
                self.graph[name] = [dep]

    def topological_sort(self):
        visited = set()
        stack = set()

        def dfs(node):
            if node in stack:
                raise ValueError(f"Circular dependency detected at {node}.")
            if node in visited:
                return

            stack.add(node)
            for dep in self.graph.get(node, []):
                if dep in self.fields:
                    dfs(dep)
            stack.remove(node)
            visited.add(node)
            self.order.append(node)

        for field in self.fields.keys():
            dfs(field)

        return self.order

    def resolve(self):
        self.build_graph()
        return self.topological_sort()


# ============================================================
# 2. Value Generator
# ============================================================

class ValueGenerator:
    def __init__(self):
        pass
    

    def match_rule(self, x, rules):
        """
        Evaluate rules for conditional distributions.
        Supports:
        - Numeric ranges: "23-29"
        - Boolean expressions: "x >= 18 and x <= 22"
        - Exact numeric match: "18"
        - Exact string match: "male", "female", etc.
        - Default fallback
        """
        # -------- Try numeric rules
        is_numeric = False
        try:
            x_num = float(x)
            is_numeric = True
        except (ValueError, TypeError):
            pass

        for rule_key, rule_val in rules.items():
            if rule_key == "default":
                continue

            if is_numeric:
                # numeric range like "23-29"
                if re.match(r"^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$", rule_key):
                    a, b = map(float, rule_key.split("-"))
                    if a <= x_num <= b:
                        return rule_val

                # boolean expression with 'x'
                elif "x" in rule_key:
                    try:
                        if eval(rule_key, {"x": x_num}):
                            return rule_val
                    except Exception:
                        pass

                # exact numeric match
                else:
                    try:
                        if float(rule_key) == x_num:
                            return rule_val
                    except Exception:
                        pass
            else:
                # string match
                if str(rule_key) == str(x):
                    return rule_val

        # fallback to default
        return rules.get("default")
        
    
    def generate(self, field_name: str, field_info, row: dict):
        extra = field_info.json_schema_extra or {}

        # -----------------------------------------------------------
        # 1) Formula
        # -----------------------------------------------------------
        formula = extra.get("formula")
        if formula:
            try:
                if 'row' in formula:
                    return eval(formula, {"row": row, "random": random, "np": np})
                return eval(formula, {"random": random, "np": np, "uuid": uuid})
            except Exception as e:
                print(f"Formula error for field {field_name}: {e}")

        # -----------------------------------------------------------
        # 2) Distribution
        # -----------------------------------------------------------
        dist = extra.get("distribution")
        if not dist:
            return None

        dist_type = dist.get("dist")

        # ------------------------
        # CONDITIONAL LOGIC
        # ------------------------
        condition_on = dist.get("condition_on")
        rules = dist.get("rules")

        if condition_on and rules:
            cond_value = row.get(condition_on)

            # Use the new rule parser
            selected_rule = self.match_rule(cond_value, rules)

            if selected_rule:
                dist = {**dist, **selected_rule}


        # =====================================================================
        # Distribution Types
        # =====================================================================

        # -------- Normal --------
        if dist_type == "normal":
            mean = dist.get("mean")
            sd = dist.get("sd")
            min_v = dist.get("min")
            max_v = dist.get("max")
            v = np.random.normal(mean, sd)
            return int(np.clip(v, min_v, max_v))

        # -------- Poisson --------
        if dist_type == "poisson":
            min_v = dist.get("min", 0)
            lam = dist.get("lambda", 1)
            generated_value = int(np.random.poisson(lam))
            if generated_value < min_v:
                generated_value = min_v
            return generated_value

        # -------- Exponential --------
        if dist_type == "exponential":
            scale = dist.get("scale", 1)
            return int(np.random.exponential(scale))

        # -------- Lognormal --------
        if dist_type == "lognormal":
            # default: mean=0, sigma=1
            mean = dist.get("mean", 0)
            sigma = dist.get("sd", 1)
            min_v = dist.get("min")
            max_v = dist.get("max")
            v = np.random.lognormal(mean, sigma)
            v = np.clip(v, min_v, max_v)
            return int(v)

        # -------- Categorical --------
        if dist_type == "categorical":
            cats = dist.get("categories")
            probs = None
            if "probs" in dist:
                probs = list(dist["probs"].values())
                cats = list(dist["probs"].keys())
            return np.random.choice(cats, p=probs) if probs else np.random.choice(cats)

        # -------- Bernoulli --------
        if dist_type == "bernoulli":
            return bool(np.random.rand() < dist.get("p", 0.5))

        return None


# ============================================================
# 3. Dataset Generator
# ============================================================

class DatasetGenerator:
    def __init__(self, schema_cls, n_rows=1000, csv_path="generated_data.csv"):
        self.schema_cls = schema_cls
        self.n_rows = n_rows
        self.csv_path = csv_path
        self.resolver = DependencyResolver(schema_cls)
        self.order = self.resolver.resolve()
        self.value_gen = ValueGenerator()

    def generate(self):
        rows = []

        for _ in range(self.n_rows):
            row = {}
            for field_name in self.order:
                field_info = self.schema_cls.model_fields[field_name]
                row[field_name] = self.value_gen.generate(field_name, field_info, row)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        print(f"Dataset saved to: {self.csv_path}")
        return df


# ============================================================
# 4. Run end-to-end generation
# ============================================================

if __name__ == "__main__":
    generator = DatasetGenerator(CustomerChurnSchema, n_rows=500, csv_path="customer_churn_synthetic.csv")
    df = generator.generate()
    print(df.head())

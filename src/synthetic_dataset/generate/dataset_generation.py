import csv
import random
import uuid
import math
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, get_origin, get_args


#TODO:
#1. pydantic schema dependencies iterator
#2. iterate over fields, check for distribution/formula/dependencies and generate value
#3. append to dataframe
# 4. save the csv 

class DataGenerator:
    """
    Extremely simple automatic data generator that:
    - Reads schema fields
    - Resolves dependencies automatically
    - Applies formulas
    - Applies distributions and conditional rules
    """

    def __init__(self, model: BaseModel):
        self.model = model
        # Support both pydantic v1 and v2 field storages:
        # - v1 exposes `__fields__` mapping to ModelField objects (which have .field_info)
        # - v2 exposes `model_fields` mapping to FieldInfo objects (which have .extra)
        # Prefer pydantic v2 attribute first to avoid deprecation warnings
        if hasattr(model, "model_fields"):
            self.fields = model.model_fields
        elif hasattr(model, "__fields__"):
            self.fields = model.__fields__
        else:
            # fallback: try to treat model as instance
            self.fields = getattr(type(model), "__fields__", getattr(type(model), "model_fields", {}))

    # -----------------------------
    #   DISTRIBUTION SAMPLERS
    # -----------------------------
    def sample_distribution(self, dist_cfg: Dict[str, Any], context: Dict[str, Any]):
        if dist_cfg is None:
            return None

        dist = dist_cfg.get("dist")

        if dist == "normal":
            mean = dist_cfg.get("mean", 0)
            sd = dist_cfg.get("sd", 1)
            val = np.random.normal(mean, sd)

            # Respect optional min/max if included
            if "min" in dist_cfg:
                val = max(val, dist_cfg["min"])
            if "max" in dist_cfg:
                val = min(val, dist_cfg["max"])
            return val

        if dist == "lognormal":
            return float(np.random.lognormal(mean=1, sigma=1))

        if dist == "poisson":
            lam = dist_cfg.get("lambda", 1)
            return int(np.random.poisson(lam))

        if dist == "exponential":
            scale = dist_cfg.get("scale", 1)
            return float(np.random.exponential(scale))

        if dist == "categorical":
            categories = dist_cfg.get("categories")
            rules = dist_cfg.get("rules", {})
            cond = dist_cfg.get("condition_on")

            if cond and cond in context:
                cond_val = context[cond]
                # find rule key for ranges (e.g. "18-22")
                for key, rule in rules.items():
                    if key == "default":
                        continue
                    if "-" in key:
                        lo, hi = key.split("-")
                        if int(lo) <= int(cond_val) <= int(hi):
                            probs = rule["probs"]
                            return random.choices(list(probs.keys()), list(probs.values()))[0]
                    else:
                        if str(cond_val) == key:
                            probs = rule["probs"]
                            return random.choices(list(probs.keys()), list(probs.values()))[0]

                # fallback
                if "default" in rules:
                    probs = rules["default"]["probs"]
                    return random.choices(list(probs.keys()), list(probs.values()))[0]

            # unconditional categorical
            return random.choice(categories)

        if dist == "bernoulli":
            p = dist_cfg.get("p", 0.5)
            return random.random() < p

        return None

    # -----------------------------
    #   MAIN GENERATION FUNCTION
    # -----------------------------
    def generate_one(self) -> Dict[str, Any]:
        obj = {}
        unresolved = set(self.fields.keys())

        # Resolve values in dependency order
        while unresolved:
            progress = False

            for name in list(unresolved):
                field = self.fields[name]

                # Normalize access to extra metadata and default across pydantic versions
                if hasattr(field, "field_info"):
                    # pydantic v1: ModelField
                    extras = getattr(field.field_info, "extra", {}) or {}
                    default_val = getattr(field, "default", None)
                else:
                    # pydantic v2: FieldInfo
                    extras = getattr(field, "extra", {}) or {}
                    default_val = getattr(field, "default", None)

                dist_cfg = extras.get("distribution")
                formula = extras.get("formula")
                deps = extras.get("dependent_on")

                # dependencies not resolved yet
                if deps:
                    if isinstance(deps, str):
                        deps = [deps]
                    # Only consider dependencies that are actually fields in the schema.
                    # If a dependency refers to an external or missing field (e.g. signup_date),
                    # don't block generation on it.
                    relevant_deps = [d for d in deps if d in self.fields]
                    if any(dep not in obj for dep in relevant_deps):
                        continue

                # -----------------------------
                # 1. formula-based generator
                # -----------------------------
                if formula:
                    obj[name] = eval(formula)
                    unresolved.remove(name)
                    progress = True
                    continue

                # -----------------------------
                # 2. literal type handling (bool, int, float, str)
                # -----------------------------
                if dist_cfg:
                    val = self.sample_distribution(dist_cfg, obj)
                    obj[name] = val
                    unresolved.remove(name)
                    progress = True
                    continue

                # -----------------------------
                # 3. default if nothing else applies
                # -----------------------------
                # If a default value was provided (including explicit None) or
                # the field has no distribution/formula and no relevant deps,
                # treat it as resolved by assigning the default (may be None).
                has_explicit_default = (default_val is not None) or (
                    hasattr(field, "required") and not getattr(field, "required")
                )
                if has_explicit_default or (not dist_cfg and not formula and not deps):
                    obj[name] = default_val
                    unresolved.remove(name)
                    progress = True
                    continue

            if not progress:
                # prevent endless loop — provide debug info to help identify blocked fields
                print("No progress in generation loop. Unresolved fields:", unresolved)
                for name in unresolved:
                    f = self.fields[name]
                    if hasattr(f, "field_info"):
                        ex = getattr(f.field_info, "extra", {}) or {}
                        dval = getattr(f, "default", None)
                    else:
                        ex = getattr(f, "extra", {}) or {}
                        dval = getattr(f, "default", None)
                    deps_here = ex.get("dependent_on")
                    print(name, "-> has_distribution=", bool(ex.get("distribution")), "formula=", ex.get("formula"), "default=", dval, "dependent_on=", deps_here)
                raise RuntimeError("Circular dependency or unresolved field detected.")

        return obj

    # -----------------------------
    #   GENERATE MANY + SAVE CSV
    # -----------------------------
    def generate(self, n: int, csv_path: str):
        rows = [self.generate_one() for _ in range(n)]
        fieldnames = list(rows[0].keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Generated {n} rows → {csv_path}")


# -----------------------------
# USAGE EXAMPLE
# -----------------------------
if __name__ == "__main__":
    from dataset_schema_definition import CustomerChurnSchema

    gen = DataGenerator(CustomerChurnSchema)
    gen.generate(n=100, csv_path="customers.csv")

import pandas as pd
import numpy as np
import uuid
from typing import get_type_hints
from pydantic import BaseModel, Field
from dataset_schema_definition import CustomerChurnSchema


class DependencyIterator:
    def __init__(self, schema_cls):
        self.schema_cls = schema_cls
        self.fields = schema_cls.model_fields  
        self.visited = set()
        self.order = []
    
    def _visit(self, field_name):
        if field_name in self.visited:
            return
        field_info = self.fields[field_name]
        extra = getattr(field_info, "extra", {})
        dep = extra.get("dependent_on")
        
        if dep:
            if isinstance(dep, list):
                for d in dep:
                    self._visit(d)
            else:
                self._visit(dep)
        
        self.visited.add(field_name)
        self.order.append(field_name)
    
    def get_order(self):
        for field_name in self.fields.keys():
            self._visit(field_name)
        return self.order

class DatasetGenerator:
    def __init__(self, schema_cls, n_rows=10):
        self.schema_cls = schema_cls
        self.n_rows = n_rows
        self.dependency_iterator = DependencyIterator(schema_cls)
        self.generation_order = self.dependency_iterator.get_order()
    
    def generate_value(self, field_info, current_row):
        """Generate value based on field_info metadata"""
        # Formula first
        formula = getattr(field_info, "formula", None)
        if formula:
            try:
                return eval(formula)
            except Exception as e:
                print(f"Error in formula {formula}: {e}")
                return None

        # Distribution handling
        print(field_info.metadata)
        extra = getattr(field_info, "metadata", {})  # <- contains distribution, dependent_on, formula
        print(extra)
        dist = extra.get("distribution")
        formula = extra.get("formula")
        if not dist:
            print(f'No distribution found for field{field_info}')
            return None

        if dist['dist'] == 'normal':
            mean = dist.get('mean', 0)
            sd = dist.get('sd', 1)
            min_val = dist.get('min', -np.inf)
            max_val = dist.get('max', np.inf)
            val = np.random.normal(mean, sd)
            return min(max(val, min_val), max_val)

        if dist['dist'] == 'poisson':
            lam = dist.get('lambda', 1)
            return np.random.poisson(lam)

        if dist['dist'] == 'categorical':
            categories = dist.get('categories')
            if categories:
                return np.random.choice(categories)
            return None

        if dist['dist'] == 'bernoulli':
            p = dist.get('p', 0.5)
            return np.random.rand() < p

        return None

    def generate_dataset(self):
        data = []

        for i in range(self.n_rows):
            row = {}
            for field_name in self.generation_order:
                field_info = self.schema_cls.model_fields[field_name] 
                row[field_name] = self.generate_value(field_info, row)
            data.append(row)

        df = pd.DataFrame(data)
        return df

if __name__ == "__main__":
    generator = DatasetGenerator(CustomerChurnSchema, n_rows=1000)
    df = generator.generate_dataset()
    print(df.head())
    # df.to_csv("llm_generated_customers.csv", index=False)


# def generate_value(field_info, current_row):
#     """Generate value based on field_info metadata"""
#     # Formula first
#     formula = getattr(field_info, "formula", None)
#     if formula:
#         try:
#             return eval(formula)
#         except Exception as e:
#             print(f"Error in formula {formula}: {e}")
#             return None

#     # Distribution handling
#     dist = getattr(field_info, "distribution", None)
#     if not dist:
#         return None

#     if dist['dist'] == 'normal':
#         mean = dist.get('mean', 0)
#         sd = dist.get('sd', 1)
#         min_val = dist.get('min', -np.inf)
#         max_val = dist.get('max', np.inf)
#         val = np.random.normal(mean, sd)
#         return min(max(val, min_val), max_val)

#     if dist['dist'] == 'poisson':
#         lam = dist.get('lambda', 1)
#         return np.random.poisson(lam)

#     if dist['dist'] == 'categorical':
#         categories = dist.get('categories')
#         if categories:
#             return np.random.choice(categories)
#         return None

#     if dist['dist'] == 'bernoulli':
#         p = dist.get('p', 0.5)
#         return np.random.rand() < p

#     return None

# def resolve_dependencies(schema_cls):
#     """Resolve field dependencies using Pydantic v2 API"""
#     fields = schema_cls.model_fields  # <-- v2 replacement for __fields__
#     sorted_fields = []
#     remaining = set(fields.keys())

#     while remaining:
#         progress = False
#         for f in list(remaining):
#             # Access extra metadata
#             extra = getattr(fields[f], "extra", {})
#             dep = extra.get("dependent_on")
            
#             if dep is None:
#                 sorted_fields.append(f)
#                 remaining.remove(f)
#                 progress = True
#             elif isinstance(dep, list) and all(d in sorted_fields for d in dep):
#                 sorted_fields.append(f)
#                 remaining.remove(f)
#                 progress = True
#             elif isinstance(dep, str) and dep in sorted_fields:
#                 sorted_fields.append(f)
#                 remaining.remove(f)
#                 progress = True

#         if not progress:
#             raise ValueError(f"Circular or unresolved dependency detected among: {remaining}")

#     return sorted_fields


# def generate_dataset(schema_cls, n_rows=10):
#     order = resolve_dependencies(schema_cls)
#     data = []

#     for i in range(n_rows):
#         row = {}
#         for field_name in order:
#             field_info = schema_cls.model_fields[field_name] 
#             row[field_name] = generate_value(field_info, row)
#         data.append(row)

#     df = pd.DataFrame(data)
#     return df

# # ---------------------------
# # Usage example
# # ---------------------------
# # Replace `CustomerChurnSchema` with your full Pydantic model
# # from your file
# # from your_module import CustomerChurnSchema

# df = generate_dataset(CustomerChurnSchema, n_rows=10)
# print(df.head())
# df.to_csv("output.csv", index=False)

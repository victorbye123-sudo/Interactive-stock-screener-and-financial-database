import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
n_rows = 1000

# Possible values for genomic and clinical fields
genes = ['BRCA1', 'BRCA2', 'TP53', 'EGFR', 'KRAS']
variant_types = ['missense', 'nonsense', 'frameshift', 'splice_site', 'silent', 'none']
clinical_diagnoses = ['Breast Cancer', 'Lung Cancer', 'Colorectal Cancer', 'Healthy', 'Ovarian Cancer']
treatment_types = ['Chemotherapy', 'Surgery', 'Radiation Therapy', 'Targeted Therapy', 'Immunotherapy', 'None']
outcomes = ['Remission', 'Progression', 'Stable', 'Deceased']

def random_variant_position():
    # Position on gene typically between 1 and ~10000 base pairs (simplified)
    return random.randint(1, 10000)

def random_diagnosis_date():
    return fake.date_between(start_date='-10y', end_date='today')

def random_treatment_start(diagnosis_date):
    # Treatment usually starts within 0-180 days after diagnosis
    start_offset = np.random.randint(0, 181)
    return pd.to_datetime(diagnosis_date) + pd.Timedelta(days=start_offset)

data = {
    'PatientID': [f'P{i:05d}' for i in range(1, n_rows + 1)],
    'Age': [np.random.randint(18, 90) for _ in range(n_rows)],
    'Sex': [random.choice(['M', 'F']) for _ in range(n_rows)],
    'Gene': [random.choice(genes) for _ in range(n_rows)],
    'Variant_Type': [random.choices(variant_types, weights=[0.2, 0.15, 0.1, 0.05, 0.4, 0.1])[0] for _ in range(n_rows)],
    'Variant_Position': [random_variant_position() for _ in range(n_rows)],
    'Clinical_Diagnosis': [random.choices(clinical_diagnoses, weights=[0.25, 0.2, 0.15, 0.3, 0.1])[0] for _ in range(n_rows)],
    'Diagnosis_Date': [random_diagnosis_date() for _ in range(n_rows)],
    'Treatment_Type': [],
    'Treatment_Start_Date': [],
    'Outcome': [random.choices(outcomes, weights=[0.5, 0.2, 0.25, 0.05])[0] for _ in range(n_rows)]
}

# Generate treatment info conditional on diagnosis and outcome
for diag_date, outcome in zip(data['Diagnosis_Date'], data['Outcome']):
    if outcome == 'Deceased' or random.random() < 0.1:
        treatment = 'None'
        treatment_start = None
    else:
        treatment = random.choice(treatment_types[:-1])  # exclude 'None' for treated patients
        treatment_start = random_treatment_start(diag_date)
    data['Treatment_Type'].append(treatment)
    data['Treatment_Start_Date'].append(treatment_start)

df = pd.DataFrame(data)

print(df.head())
df.to_csv("synthetic_genomic_clinical_data.csv", index=False)
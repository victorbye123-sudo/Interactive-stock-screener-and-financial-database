import pandas as pd
import numpy as np
from faker import Faker

# Initialize faker
fake = Faker()

# Set how many rows of data you want
n_rows = 1000

# Generate synthetic data
data = {
    'CustomerID': [i for i in range(1, n_rows + 1)],
    'Name': [fake.name() for _ in range(n_rows)],
    'Email': [fake.email() for _ in range(n_rows)],
    'Age': [np.random.randint(18, 70) for _ in range(n_rows)],
    'SignupDate': [fake.date_between(start_date='-5y', end_date='today') for _ in range(n_rows)],
    'Country': [fake.country() for _ in range(n_rows)],
    'PurchaseAmount': [round(np.random.exponential(scale=100), 2) for _ in range(n_rows)],
}

# Create DataFrame
df = pd.DataFrame(data)

# View the first few rows
print(df.head())

# Optional: Save to CSV
df.to_csv("synthetic_customers.csv", index=False)

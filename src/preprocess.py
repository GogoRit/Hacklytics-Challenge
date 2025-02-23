import pandas as pd
import numpy as np

# Load Dataset
file_path = '/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/processed/HazardMitigationAssistanceProjects.csv'
df = pd.read_csv(file_path)

# Step 1: Calculate Project Timeline in Months Using dateInitiallyApproved
df['dateInitiallyApproved'] = pd.to_datetime(df['dateInitiallyApproved'], errors='coerce')
df['dateClosed'] = pd.to_datetime(df['dateClosed'], errors='coerce')
df['ProjectTimeline(in months)'] = np.round((df['dateClosed'] - df['dateInitiallyApproved']).dt.days / 30)

# Step 2: Extract Required Columns
df = df[['state', 'federalShareObligated', 'benefitCostRatio', 'ProjectTimeline(in months)']]
df.rename(columns={'federalShareObligated': 'FederalFunding', 'benefitCostRatio': 'benefitCostRatio'}, inplace=True)

# Step 3: Drop Rows with Null Values
df = df.dropna()

# Step 4: Assign Zone Risk Level
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'Frequency']
state_counts['ZoneRiskLevel'] = pd.cut(
    state_counts['Frequency'],
    bins=[0, state_counts['Frequency'].quantile(0.33), state_counts['Frequency'].quantile(0.66), state_counts['Frequency'].max()],
    labels=['Low', 'Moderate', 'High'],
    include_lowest=True
)
df = df.merge(state_counts[['state', 'ZoneRiskLevel']], on='state', how='left')

# Output Enhanced Dataset
df.to_csv('/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/processed/Enhanced_HazardMitigationDataset.csv', index=False)
df.head()

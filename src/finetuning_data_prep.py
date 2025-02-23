import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# Load Dataset
file_path = '/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/processed/Enhanced_HazardMitigationDataset.csv'
df = pd.read_csv(file_path)

# Step 1: Data Preprocessing
df = df.dropna()

# Step 2: Normalize Numeric Data
scaler = MinMaxScaler()
df[['FederalFunding', 'benefitCostRatio', 'ProjectTimeline(in months)']] = scaler.fit_transform(
    df[['FederalFunding', 'benefitCostRatio', 'ProjectTimeline(in months)']]
)

# Step 3: Assign Numeric Values to ZoneRiskLevel
df['ZoneRiskLevelNumeric'] = df['ZoneRiskLevel'].map({'Low': 0.2, 'Moderate': 0.5, 'High': 0.8})

# Step 4: Calculate Composite Risk Score with ZoneRiskLevel
w1, w2, w3, w4 = 0.4, 0.25, 0.2, 0.15

df['RiskScore'] = (
    w1 * df['FederalFunding'] +
    w2 * (1 / df['benefitCostRatio'].replace(0, 1)) +
    w3 * df['ProjectTimeline(in months)'] +
    w4 * df['ZoneRiskLevelNumeric']
)

# Step 5: Apply Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
df['InsuranceZone'] = gmm.fit_predict(df[['RiskScore']])

# Map Clusters
zone_map = {
    df['RiskScore'].nlargest(len(df)//3).index[i]: 'High' for i in range(len(df)//3)
}
zone_map.update({
    df['RiskScore'].nsmallest(len(df)//3).index[i]: 'Low' for i in range(len(df)//3)
})
df['InsuranceZone'] = df.index.map(zone_map).fillna('Moderate')

# Step 6: Randomly Sample 5000 Data Points
df_sampled = df.sample(n=5000, random_state=42)

# Step 7: Save as CSV for Gemini Playground
df_sampled.to_csv('/Users/gaurankmaheshwari/Documents/Projects/hacklytics/data/fine_tuning_dataset/gemini_fine_tune_5000.csv', index=False)

print("Random 5000 data points saved as 'gemini_fine_tune_5000.csv' for Gemini fine-tuning.")
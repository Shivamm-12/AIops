import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('devops_tool_recommendations.csv')

# Select the continuous features to standardize
continuous_features = ['Project Size', 'Project Length']

# Step 1: Plot histograms for continuous features before standardization
plt.figure(figsize=(12, 6))

for i, feature in enumerate(continuous_features, 1):
    plt.subplot(1, 2, i)
    df[feature].hist(bins=20, alpha=0.7)
    plt.title(f'Before Standardization: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 2: Apply standardization using StandardScaler
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[continuous_features] = scaler.fit_transform(df[continuous_features])

# Step 3: Plot histograms for continuous features after standardization
plt.figure(figsize=(12, 6))

for i, feature in enumerate(continuous_features, 1):
    plt.subplot(1, 2, i)
    df_standardized[feature].hist(bins=20, alpha=0.7)
    plt.title(f'After Standardization: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 4: Display dataset specification (overview)
print("Dataset Specification:")
print(df.describe())  # Provides basic statistics including mean and std dev

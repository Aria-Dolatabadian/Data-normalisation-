import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

# Read data from CSV file
hw_df = pd.read_csv('SOCR-HeightWeight.csv')

# Display descriptive statistics of the original data
original_data_description = hw_df.describe()
print("Original Data Description:")
print(original_data_description)

# Scale the 'Height' and 'Weight' columns using Min-Max scaling
hw_scaled = minmax_scale(hw_df[['Height(Inches)', 'Weight(Pounds)']], feature_range=(0, 1))

# Add normalized columns to the DataFrame
hw_df['Height(Norm)'] = hw_scaled[:, 0]
hw_df['Weight(Norm)'] = hw_scaled[:, 1]

# Display descriptive statistics of the normalized data
normalized_data_description = hw_df.describe()
print("\nNormalized Data Description:")
print(normalized_data_description)

# Save original and normalized data descriptions to CSV files
original_description_file_path = 'original_data_description.csv'
normalized_description_file_path = 'normalized_data_description.csv'

original_data_description.to_csv(original_description_file_path)
normalized_data_description.to_csv(normalized_description_file_path)

print(f"\nOriginal data description exported to '{original_description_file_path}'")
print(f"Normalized data description exported to '{normalized_description_file_path}'")

# Save normalized data to a new CSV file
normalized_file_path = 'normalized_height_weight_data.csv'
hw_df.to_csv(normalized_file_path, index=False)
print(f"\nNormalized data exported to '{normalized_file_path}'")

# Visualize histograms of both original and normalized data
plt.figure(figsize=(10, 10))
hw_df[['Height(Inches)', 'Height(Norm)', 'Weight(Pounds)', 'Weight(Norm)']].hist()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Added seaborn for kernel density estimation (KDE)

# Import data from CSV file
file_path = 'non_normal_data.csv'  # Replace 'non_normal_data.csv' with the actual file path
original_data = pd.read_csv(file_path, header=None).values.flatten()

# Visualize original data with KDE curve
plt.subplot(2, 1, 1)
sns.histplot(original_data, bins=30, kde=True, color='blue')
plt.title('Original Data Distribution with KDE')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Z-score normalization
z_score_normalized_data = (original_data - np.mean(original_data)) / np.std(original_data)

# Visualize normalized data with KDE curve
plt.subplot(2, 1, 2)
sns.histplot(z_score_normalized_data, bins=30, kde=True, color='green')
plt.title('Z-Score Normalized Data Distribution with KDE')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Export normalized data to a new CSV file
normalized_file_path = 'z_score_normalized_data.csv'
np.savetxt(normalized_file_path, z_score_normalized_data, delimiter=',')

print("Z-score normalized data exported to 'z_score_normalized_data.csv'")

# Show plots
plt.show()

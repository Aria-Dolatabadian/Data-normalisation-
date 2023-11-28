import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# Import data from CSV
imported_data = pd.read_csv('original_data.csv')

# Check for non-positive values
non_positive_columns = imported_data.columns[(imported_data <= 0).any()]
if not non_positive_columns.empty:
    print(f"Warning: Non-positive values found in columns {non_positive_columns}. Shifting the data.")

    # Shift non-positive values to make them positive
    imported_data[non_positive_columns] += np.abs(imported_data[non_positive_columns].min()) + 0.1

# Visualize the original data
imported_data.hist(alpha=0.5, color='blue', edgecolor='black', bins=20)
plt.suptitle('Original Data')
plt.show()

# Perform Box-Cox transformation
transformer = PowerTransformer(method='box-cox', standardize=True)
transformed_data = transformer.fit_transform(imported_data)

# Visualize the transformed data
transformed_data_df = pd.DataFrame(transformed_data, columns=imported_data.columns)
transformed_data_df.hist(alpha=0.5, color='green', edgecolor='black', bins=20)
plt.suptitle('Normalized Data')
plt.show()

# Export transformed data to CSV
transformed_data_df.to_csv('transformed_data.csv', index=False)

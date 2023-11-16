import pandas as pd

df = pd.read_csv('data/cardiovascular_data-original-train.csv')

# Determine the number of rows in the DataFrame
num_rows = len(df)

# Decide the fraction of data you want in each file
fraction = 0.33333333333  # data percentage in each file

# Calculate the number of rows that should be in each file
split_size = round(num_rows * fraction)

# Use a loop to split the DataFrame into smaller DataFrames
for i in range(0, num_rows, split_size):
    df_subset = df.iloc[i:i+split_size]
    df_subset.to_csv(f'data/split_file_{i//split_size + 1}.csv', index=False)
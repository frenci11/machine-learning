import numpy as np

# Create a 4D array with shape (2, 3, 4, 5)
array = np.arange(2*3*4*5).reshape(2, 3, 4, 5)

# Select the first batch
first_batch = array[0]
print("First batch shape:", first_batch.shape)

# Select the first row of the first batch
first_row_of_first_batch = array[0, 0]
print("First row of first batch shape:", first_row_of_first_batch.shape)

# Select the first column of the first row of the first batch
first_column_of_first_row_of_first_batch = array[0, 0, 0]
print("First column of first row of first batch shape:", first_column_of_first_row_of_first_batch.shape)

# Select the first channel of the first column of the first row of the first batch
first_channel_of_first_column_of_first_row_of_first_batch = array[0, 0, 0, 0]
print("First channel value:", first_channel_of_first_column_of_first_row_of_first_batch)

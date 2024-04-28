import numpy as np

# Define the sets
set1 = [0, 8, 12, 20, 10, 25, 27, 38, 57]
set2 = [8, 9, 11, 12, 22, 33, 44, 55, 66]

# Calculate the standard deviation for each set
std_set1 = np.std(set1, ddof=1)  # Using sample standard deviation (ddof=1)
std_set2 = np.std(set2, ddof=1)

# Find which set has a larger spread
larger_spread_set = "Set1" if std_set1 > std_set2 else "Set2"
print(std_set1)
# Calculate the covariance for the set with larger spread
if larger_spread_set == "Set1":
    # Create a second set for covariance computation (can be any related set)
    set1_additional = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    covariance = np.cov(set1, set1_additional)[0, 1]
elif larger_spread_set == "Set2":
    set2_additional = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    covariance = np.cov(set2, set2_additional)[0, 1]

print(std_set1, std_set2, larger_spread_set, covariance)  # Return the results

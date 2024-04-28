import numpy as np

# Define the matrix A
A = np.array([[7, 3], [3, -1]])

# Calculate the eigenvalues and eigenvectors of the matrix A
eigenvalues, eigenvectors = np.linalg.eig(A)

# Display the eigenvalues and eigenvectors
print(eigenvalues, eigenvectors)

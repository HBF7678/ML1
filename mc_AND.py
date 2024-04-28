# Define the inputs for the AND gate
inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# Weights and threshold for the AND gate
weights = (1, 1)  # both weights are 1
threshold = 2  # threshold for AND gate

# Define the activation function
def activation(x):
    return 1 if x >= threshold else 0

# Calculate outputs for each input combination
outputs = []
for inp in inputs:
    # Compute the linear combination of inputs with weights
    linear_combination = sum(w * x for w, x in zip(weights, inp))
    # Apply the activation function
    output = activation(linear_combination)
    outputs.append((inp, output))

print(outputs)  # Output the inputs with corresponding outputs

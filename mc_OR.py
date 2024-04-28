# Define the inputs for the OR gate
inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# Weights and threshold for the OR gate
weights = (1, 1)  # both inputs should be considered
threshold = 1  # to achieve OR behavior

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

print(outputs)  # Output the results for the OR gate

from TinyGrad import Value
from NeuralNetwork import MLP

# Create a 2-layer MLP with 3 inputs → 4 → 1 output
model = MLP(3, [4, 1])
print(model)

# Forward pass
x = [Value(1.0), Value(2.0), Value(3.0)]
output = model(x)
print("Output:", output)

# Before backward pass
print(model.parameters())

# Backward pass
output.backward()

# After backward pass
print(model.parameters())

# Update parameters (simplified example)
for p in model.parameters():
    p.data -= 0.01 * p.grad

# Updated parameters
print(model.parameters())


# After zero_grad
model.zero_grad()
print(model.parameters())

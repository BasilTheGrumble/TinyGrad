import random
from TinyGrad import Value


class Module:
    """
    Base class for all neural network modules.
    Provides common functionality for parameter management.
    """

    def zero_grad(self):
        """
        Resets gradients of all parameters to zero.
        This should be called before each backward pass.
        """

        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Returns a list of all trainable parameters in the module.
        This base implementation returns an empty list.
        """

        return []


class Neuron(Module):
    """
    A single artificial neuron with optional non-linear activation.
    """

    def __init__(self, input_size, nonlin=True):
        """
        Initialize a neuron with given input size and activation.

        Args:
            input_size: Number of input features
            nonlin: Whether to apply ReLU activation (default: True)
        """

        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Compute the neuron's output for the given input.

        Args:
            x: List of input values (must match input_size)

        Returns:
            Activation value after applying weights, bias, and non-linearity
        """

        activation = sum((w * xi for w, xi in zip(self.weights, x)), self.bias)

        return activation.relu() if self.nonlin else activation

    def parameters(self):
        """Return all trainable parameters (weights + bias)"""

        return self.weights + [self.bias]

    def __repr__(self):
        """String representation showing neuron type and input size"""

        activation = "ReLU" if self.nonlin else "Linear"

        return f"{activation}Neuron(input_size={len(self.weights)})"


class Layer(Module):
    """
    A layer of multiple neurons operating in parallel.
    """

    def __init__(self, input_size, output_size, **kwargs):
        """
        Create a fully connected layer.

        Args:
            input_size: Number of input features
            output_size: Number of neurons in the layer
            **kwargs: Additional arguments passed to Neuron constructor
        """

        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]

    def __call__(self, x):
        """
        Compute layer output for given input.

        Args:
            x: Input values (list of numbers)

        Returns:
            Single value if output_size=1, else list of neuron outputs
        """

        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        """Return all parameters from all neurons in the layer"""

        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        """Show layer configuration with all neurons"""

        return f"Layer([{', '.join(str(neuron) for neuron in self.neurons)}])"


class MLP(Module):
    """
    Multi-Layer Perceptron (feedforward neural network)
    """

    def __init__(self, input_size, layer_sizes):
        """
        Create MLP with given architecture.

        Args:
            input_size: Number of input features
            layer_sizes: List defining the number of neurons in each layer
        """

        sizes = [input_size] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            # Create layers with ReLU except for last layer
            layer = Layer(sizes[i], sizes[i + 1], nonlin=(i != len(layer_sizes) - 1))
            self.layers.append(layer)

    def __call__(self, x):
        """
        Perform forward pass through the network.

        Args:
            x: Input values (list of numbers)

        Returns:
            Output of the last layer
        """

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        """Return all trainable parameters in the network"""

        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        """Show network architecture with all layers"""

        return f"MLP(layers=[{', '.join(str(layer) for layer in self.layers)}])"
    
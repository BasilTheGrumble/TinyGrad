import math


class Value:
    """
    The Value class represents a scalar value with the ability to automatically compute gradients.
    It is used to build a computational graph and perform backpropagation.
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initialize a Value object.
        :param data: numerical value (scalar).
        :param _children: set of parent nodes in the computational graph.
        :param _op: string describing the operation that created this node.
        """

        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """
        Overload the addition operator (+).
        :param other: second operand (can be a number or a Value object).
        :return: a new Value object representing the result of addition.
        """

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Overload the multiplication operator (*).
        :param other: second operand (can be a number or a Value object).
        :return: a new Value object representing the result of multiplication.
        """

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        """
        Overload the division operator (/).
        :param other: second operand (a number object).
        :return: a new Value object representing the result of division.
        """

        out = self * other ** -1

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= out.grad * self.data / (other.data ** 2)

        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Overload the exponentiation operator (**).

        Allows raising a Value object to the power of another Value object or a number.
        Automatically computes gradients for backpropagation of errors.

        :param other: The exponent (can be a number or a Value object).
        :return: A new Value object containing the result of the operation.
        """

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')

        def _backward():
            a = self.data
            b = other.data
            grad_a = b * (a ** (b - 1)) if a != 0 else 0.0
            grad_b = (a ** b) * math.log(a) if a > 0 else 0.0

            self.grad += grad_a * out.grad
            other.grad += grad_b * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """
        ReLU (Rectified Linear Unit) activation function.
        :return: a new Value object representing the result of applying ReLU.
        """

        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        """
        Sigmoid activation function.
        :return: a new Value object representing the result of applying Sigmoid.
        """

        out = Value(1 / (1 + (-self).exp().data), (self,), 'Sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """
        Hyperbolic tangent (tanh) activation function.
        :return: a new Value object representing the result of applying tanh.
        """

        out = Value((2 / (1 + (-2 * self).exp().data)) - 1, (self,), 'Tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def leaky_relu(self, alpha=0.01):
        """
        Leaky ReLU activation function.
        :param alpha: slope coefficient for negative values.
        :return: a new Value object representing the result of applying Leaky ReLU.
        """

        out = Value(alpha * self.data if self.data < 0 else self.data, (self,), f'LeakyReLU({alpha})')

        def _backward():
            self.grad += (alpha if self.data < 0 else 1) * out.grad

        out._backward = _backward

        return out

    def elu(self, alpha=1.0):
        """
        ELU (Exponential Linear Unit) activation function.
        :param alpha: coefficient for the exponential part.
        :return: a new Value object representing the result of applying ELU.
        """

        out = Value(
            alpha * ((self).exp().data - 1) if self.data < 0 else self.data,
            (self,), f'ELU({alpha})'
        )

        def _backward():
            self.grad += (alpha * out.exp().data if self.data < 0 else 1) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        """
        Exponential function (e^x).
        :return: a new Value object representing the result of raising e to the power of self.data.
        """

        out = Value(math.exp(self.data), (self,), 'Exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Perform backpropagation to compute gradients.
        """

        topo = []
        visited = set()

        def build_topo(v):
            """
            Build the topological order of nodes in the computational graph.
            Topological sorting ensures that parent nodes are processed before their children.
            :param v: current node in the graph.
            """
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1  # Gradient of the final node is 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        """Overload unary negation (-self)."""

        return self * -1

    def __sub__(self, other):
        """Overload subtraction operator (self - other)."""

        return self + (-other)

    def __radd__(self, other):
        """Overload right addition (other + self)."""

        return self + other

    def __rmul__(self, other):
        """Overload right multiplication (other * self)."""

        return self * other

    def __repr__(self):
        """String representation of the object."""

        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

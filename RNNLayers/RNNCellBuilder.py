from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers


class RNNCellBuilder(Layer):
    def __init__(self, units: int,
                 states: list,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 use_bias: bool = True,
                 **kwargs):

        super(RNNCellBuilder, self).__init__(**kwargs)

        self.units = units
        self.kernel_activation = activations.get(kernel_activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

        self.states = states
        self.states_number = len(states)

        self.vars = {}
        self.w = {} # Weights with name

        self._reccurent_list = []
        self._kernel_list = []
        self._operation_list = []

    def add_recurrent(self, name: str, inputs: list):
        self._reccurent_list.append([name, inputs])
        self._operation_list.append([name, "r"])

        f = self._generate_func(inputs, name, self.recurrent_activation)

        self.vars[name] = [inputs, f]

        return self

    def add_kernel(self, name: str, inputs: list):
        self._kernel_list.append([name, inputs])
        self._operation_list.append([name, "k"])

        f = self._generate_func(inputs, name, self.kernel_activation)

        self.vars[name] = [inputs, f]

        return self

    def _generate_func(self, inputs: list, name: str, activation):
        def f(X: list):
            assert len(X) == len(inputs)
            z = []

            for i, n in enumerate(inputs):
                if i == 0:
                    z.append(K.dot(X[i], self.w["%s_%s" % (name, n)]))
                else:
                    z.append(K.dot(X[i], self.w["%s_%s" % (name, n)]) + z[-1])

            if self.use_bias:
                if len(z) == 0:
                    z.append(self.w["%s_bias" % name])
                else:
                    z.append(self.w["%s_bias" % name] + z[-1])

            return activation(z[-1])

        return f

    def add_var(self, name: str, inputs: list, f):
        self.vars[name] = [inputs, f]

        return self

    def build(self, input_shape):
        input_dim = input_shape[-1]

        full_list = self._reccurent_list + self._kernel_list

        for name, inputs in full_list:
            for input in inputs:
                shape = (self.units, self.units)
                if input == "X":
                    shape = (input_dim, self.units)

                initializer = self.recurrent_initializer if name in self._reccurent_list else self.kernel_initializer

                self.w["%s_%s" % (name, input)] = self.add_weight("%s_%s" % (name, input),
                                                                  shape,
                                                                  initializer=initializer)

            if self.use_bias:
                self.w["%s_bias" % name] = self.add_weight("%s_bias" % name, (self.units, ),
                                                           initializer=self.bias_initializer)

    def get_initial_state(self, inputs):
        batch_size = tf.compat.v1.shape(inputs)[0]
        dtype = inputs.dtype

        flat_dims = tf.TensorShape([self.units]).as_list()
        init_state_size = [batch_size] + flat_dims

        return [tf.zeros(init_state_size, dtype=dtype) for _ in self.states]

    def call(self, inputs, states, **kwargs):
        assert len(states) == self.states_number

        v = {"X": inputs}

        for i, state in enumerate(self.states):
            v[state] = states[i]

        for name, [input_list, f] in self.vars.items():
            v[name] = f([v[input_name] for input_name in input_list])

        next_states = []
        for state in self.states:
            assert "%s_next" % state in v

            next_states.append(v["%s_next" % state])

        return v[self.states[0]], next_states

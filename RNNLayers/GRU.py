from RNNLayers.RNNBuilder import RNNCellBuilder, AbstractRNNBuilder, AbstractBiRNNBuilder


def gru_cell(units: int,
             kernel_activation: str = "tanh",
             recurrent_activation: str = "hard_sigmoid",
             kernel_initializer: str = "glorot_uniform",
             recurrent_initializer: str = "orthogonal",
             bias_initializer: str = "zeros",
             use_bias: bool = True,
             **kwargs) -> RNNCellBuilder:

    cell = RNNCellBuilder(units, ["h"], kernel_activation, recurrent_activation,
                          kernel_initializer, recurrent_initializer, bias_initializer, use_bias, **kwargs)

    return cell \
        .add_recurrent("update", ["X", "h"]) \
        .add_recurrent("reset", ["X", "h"]) \
        .add_kernel("output", ["X", "reset"]) \
        .add_var("h_next", ["update", "h", "output"], lambda x: (1 - x[0]) * x[1] + x[0] * x[2])


class GRU(AbstractRNNBuilder):
    def __init__(self,
                 units: int,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 backward: bool = False,
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

        super(GRU, self).__init__(units, kernel_activation, recurrent_activation,
                                  kernel_initializer, recurrent_initializer, bias_initializer,
                                  backward, return_sequences, use_bias, name="GRU", **kwargs)

    def get_cell(self, units: int,
                 kernel_activation: str,
                 recurrent_activation: str,
                 kernel_initializer: str,
                 recurrent_initializer: str,
                 bias_initializer: str,
                 use_bias: bool,
                 **kwargs) -> RNNCellBuilder:

        return gru_cell(units, kernel_activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer,
                        use_bias, **kwargs)


class BiGRU(AbstractBiRNNBuilder):
    def __init__(self,
                 units: int,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

        super(BiGRU, self).__init__(units, kernel_activation, recurrent_activation,
                                    kernel_initializer, recurrent_initializer, bias_initializer,
                                    return_sequences, use_bias, name="BiGRU", **kwargs)

    def get_cell_forward(self, units: int,
                         kernel_activation: str,
                         recurrent_activation: str,
                         kernel_initializer: str,
                         recurrent_initializer: str,
                         bias_initializer: str,
                         use_bias: bool,
                         **kwargs) -> RNNCellBuilder:

        return gru_cell(units, kernel_activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer,
                        use_bias, **kwargs)

    def get_cell_backward(self, units: int,
                          kernel_activation: str,
                          recurrent_activation: str,
                          kernel_initializer: str,
                          recurrent_initializer: str,
                          bias_initializer: str,
                          use_bias: bool,
                          **kwargs) -> RNNCellBuilder:

        return gru_cell(units, kernel_activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer,
                        use_bias, **kwargs)

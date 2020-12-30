from RNNLayers.RNNBuilder import RNNCellBuilder, AbstractRNNBuilder, AbstractBiRNNBuilder


def lstm_cell(units: int,
              peephole: bool = True,
              kernel_activation: str = "tanh",
              recurrent_activation: str = "hard_sigmoid",
              kernel_initializer: str = "glorot_uniform",
              recurrent_initializer: str = "orthogonal",
              bias_initializer: str = "zeros",
              use_bias: bool = True,
              **kwargs) -> RNNCellBuilder:

    cell = RNNCellBuilder(units, ["h", "C"], kernel_activation, recurrent_activation,
                          kernel_initializer, recurrent_initializer, bias_initializer, use_bias, **kwargs)

    if peephole:
        return cell \
            .add_recurrent("input", ["X", "h", "C"]) \
            .add_recurrent("forget", ["X", "h", "C"]) \
            .add_recurrent("output", ["X", "h", "C"]) \
            .add_kernel("cell", ["X", "h"]) \
            .add_var("C_next", ["forget", "C", "input", "cell"],
                     lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
            .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])

    return cell \
        .add_recurrent("input", ["X", "h"]) \
        .add_recurrent("forget", ["X", "h"]) \
        .add_recurrent("output", ["X", "h"]) \
        .add_kernel("cell", ["X", "h"]) \
        .add_var("C_next", ["forget", "C", "input", "cell"],
                 lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
        .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])


class LSTM(AbstractRNNBuilder):
    def __init__(self,
                 units: int,
                 peephole: bool = True,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 backward: bool = False,
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

        self.peephole = peephole

        super(LSTM, self).__init__(units, kernel_activation, recurrent_activation,
                                   kernel_initializer, recurrent_initializer, bias_initializer,
                                   backward, return_sequences, use_bias, **kwargs)

    def get_cell(self, units: int,
                 kernel_activation: str,
                 recurrent_activation: str,
                 kernel_initializer: str,
                 recurrent_initializer: str,
                 bias_initializer: str,
                 use_bias: bool,
                 **kwargs) -> RNNCellBuilder:

        return lstm_cell(units, self.peephole,
                         kernel_activation, recurrent_activation,
                         kernel_initializer, recurrent_initializer, bias_initializer,
                         use_bias, **kwargs)


class BiLSTM(AbstractBiRNNBuilder):
    def __init__(self,
                 units: int,
                 peephole: bool = True,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

        self.peephole = peephole

        super(BiLSTM, self).__init__(units, kernel_activation, recurrent_activation,
                                     kernel_initializer, recurrent_initializer, bias_initializer,
                                     return_sequences, use_bias, **kwargs)

    def get_cell_forward(self, units: int,
                         kernel_activation: str,
                         recurrent_activation: str,
                         kernel_initializer: str,
                         recurrent_initializer: str,
                         bias_initializer: str,
                         use_bias: bool,
                         **kwargs) -> RNNCellBuilder:

        return lstm_cell(units, self.peephole,
                         kernel_activation, recurrent_activation,
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

        return lstm_cell(units, self.peephole,
                         kernel_activation, recurrent_activation,
                         kernel_initializer, recurrent_initializer, bias_initializer,
                         use_bias, **kwargs)

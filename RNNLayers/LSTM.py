import tensorflow as tf
from RNNLayers.RNNBuilder import RNNCellBuilder, AbstractRNNBuilder, AbstractBiRNNBuilder
import tensorflow.keras.backend as K


def lstm_cell(units: int,
             kernel_activation: str = "tanh",
             recurrent_activation: str = "sigmoid",
             kernel_initializer: str = "glorot_uniform",
             recurrent_initializer: str = "orthogonal",
             bias_initializer: str = "zeros",
             use_bias: bool = True,
             **kwargs) -> RNNCellBuilder:

    cell = RNNCellBuilder(units, ["h", "C"], kernel_activation, recurrent_activation,
                          kernel_initializer, recurrent_initializer, bias_initializer, use_bias, **kwargs)

    return cell\
        .add_recurrent("input", ["X", "h"])\
        .add_recurrent("forget", ["X", "h"])\
        .add_recurrent("output", ["X", "h"])\
        .add_kernel("cell", ["X", "h"])\
        .add_var("C_next", ["forget", "C", "input", "cell"],
                 lambda X: cell.recurrent_activation(X[0] * X[1] + X[2] * X[3]))\
        .add_var("h_next", ["C_next", "output"], lambda X: cell.kernel_activation(X[0]) * X[1])


class LSTM(AbstractRNNBuilder):
    def __init__(self,
                 units: int,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 reversed: bool = False,
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

        super(LSTM, self).__init__(units, kernel_activation, recurrent_activation,
                                  kernel_initializer, recurrent_initializer, bias_initializer,
                                  reversed, return_sequences, use_bias, **kwargs)

    def get_cell(self, units: int,
                 kernel_activation: str,
                 recurrent_activation: str,
                 kernel_initializer: str,
                 recurrent_initializer: str,
                 bias_initializer: str,
                 use_bias: bool,
                 **kwargs) -> RNNCellBuilder:

        return lstm_cell(units, kernel_activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer,
                        use_bias, **kwargs)


class BiLSTM(AbstractBiRNNBuilder):
    def __init__(self,
                 units: int,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

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

        return lstm_cell(units, kernel_activation, recurrent_activation,
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

        return lstm_cell(units, kernel_activation, recurrent_activation,
                         kernel_initializer, recurrent_initializer, bias_initializer,
                         use_bias, **kwargs)
from RNNLayers.RNNBuilder import RNNCellBuilder, AbstractRNNBuilder, AbstractBiRNNBuilder


# LSTM without Input Gate
def lstm_nig_cell(units: int,
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

    # input = 1
    if peephole:
        return cell \
            .add_recurrent("forget", ["X", "h", "C"]) \
            .add_recurrent("output", ["X", "h", "C"]) \
            .add_kernel("cell", ["X", "h"]) \
            .add_var("C_next", ["forget", "C", "cell"],
                     lambda x: cell.recurrent_activation(x[0] * x[1] + x[2])) \
            .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])

    return cell \
        .add_recurrent("forget", ["X", "h"]) \
        .add_recurrent("output", ["X", "h"]) \
        .add_kernel("cell", ["X", "h"]) \
        .add_var("C_next", ["forget", "C", "cell"],
                 lambda x: cell.recurrent_activation(x[0] * x[1] + x[2])) \
        .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])


# LSTM without Forget Gate
def lstm_nfg_cell(units: int,
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

    # forget = 1
    if peephole:
        return cell \
            .add_recurrent("input", ["X", "h", "C"]) \
            .add_recurrent("output", ["X", "h", "C"]) \
            .add_kernel("cell", ["X", "h"]) \
            .add_var("C_next", ["C", "input", "cell"],
                     lambda x: cell.recurrent_activation(x[0] + x[1] * x[2])) \
            .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])

    return cell \
        .add_recurrent("input", ["X", "h"]) \
        .add_recurrent("output", ["X", "h"]) \
        .add_kernel("cell", ["X", "h"]) \
        .add_var("C_next", ["C", "input", "cell"],
                 lambda x: cell.recurrent_activation(x[0] + x[1] * x[2])) \
        .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])


# LSTM without output Gate
def lstm_nog_cell(units: int,
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

    # output = 1
    if peephole:
        return cell \
            .add_recurrent("input", ["X", "h", "C"]) \
            .add_recurrent("forget", ["X", "h", "C"]) \
            .add_kernel("cell", ["X", "h"]) \
            .add_var("C_next", ["forget", "C", "input", "cell"],
                     lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
            .add_var("h_next", ["C_next"], lambda x: cell.kernel_activation(x[0]))

    return cell \
        .add_recurrent("input", ["X", "h"]) \
        .add_recurrent("forget", ["X", "h"]) \
        .add_kernel("cell", ["X", "h"]) \
        .add_var("C_next", ["forget", "C", "input", "cell"],
                 lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
        .add_var("h_next", ["C_next"], lambda x: cell.kernel_activation(x[0]))


# LSTM without Input Activation
def lstm_niaf_cell(units: int,
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

    # cell activation: linear
    if peephole:
        return cell \
            .add_recurrent("input", ["X", "h", "C"]) \
            .add_recurrent("forget", ["X", "h", "C"]) \
            .add_recurrent("output", ["X", "h", "C"]) \
            .add_kernel("cell", ["X", "h"], activation="linear") \
            .add_var("C_next", ["forget", "C", "input", "cell"],
                     lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
            .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])

    return cell \
        .add_recurrent("input", ["X", "h"]) \
        .add_recurrent("forget", ["X", "h"]) \
        .add_recurrent("output", ["X", "h"]) \
        .add_kernel("cell", ["X", "h"], activation="linear") \
        .add_var("C_next", ["forget", "C", "input", "cell"],
                 lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
        .add_var("h_next", ["C_next", "output"], lambda x: cell.kernel_activation(x[0]) * x[1])


# LSTM without output Activation
def lstm_noaf_cell(units: int,
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

    # h_next remove activation
    if peephole:
        return cell \
            .add_recurrent("input", ["X", "h", "C"]) \
            .add_recurrent("forget", ["X", "h", "C"]) \
            .add_recurrent("output", ["X", "h", "C"]) \
            .add_kernel("cell", ["X", "h"]) \
            .add_var("C_next", ["forget", "C", "input", "cell"],
                     lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
            .add_var("h_next", ["C_next", "output"], lambda x: x[0] * x[1])

    return cell \
        .add_recurrent("input", ["X", "h"]) \
        .add_recurrent("forget", ["X", "h"]) \
        .add_recurrent("output", ["X", "h"]) \
        .add_kernel("cell", ["X", "h"]) \
        .add_var("C_next", ["forget", "C", "input", "cell"],
                 lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
        .add_var("h_next", ["C_next", "output"], lambda x: x[0] * x[1])


def lstm_fgr_cell(units: int,
              peephole: bool = True,
              kernel_activation: str = "tanh",
              recurrent_activation: str = "hard_sigmoid",
              kernel_initializer: str = "glorot_uniform",
              recurrent_initializer: str = "orthogonal",
              bias_initializer: str = "zeros",
              use_bias: bool = True,
              **kwargs) -> RNNCellBuilder:

    cell = RNNCellBuilder(units, ["h", "C", "input", "forget", "output"], kernel_activation, recurrent_activation,
                          kernel_initializer, recurrent_initializer, bias_initializer, use_bias, **kwargs)

    if peephole:
        return cell \
            .add_recurrent("input_next", ["X", "h", "C", "input", "forget", "output"]) \
            .add_recurrent("forget_next", ["X", "h", "C", "input", "forget", "output"]) \
            .add_recurrent("output_next", ["X", "h", "C", "input", "forget", "output"]) \
            .add_kernel("cell", ["X", "h"]) \
            .add_var("C_next", ["forget_next", "C", "input_next", "cell"],
                     lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
            .add_var("h_next", ["C_next", "output_next"], lambda x: cell.kernel_activation(x[0]) * x[1])

    return cell \
        .add_recurrent("input_next", ["X", "h", "input", "forget", "output"]) \
        .add_recurrent("forget_next", ["X", "h", "input", "forget", "output"]) \
        .add_recurrent("output_next", ["X", "h", "input", "forget", "output"]) \
        .add_kernel("cell", ["X", "h"]) \
        .add_var("C_next", ["forget_next", "C", "input_next", "cell"],
                 lambda x: cell.recurrent_activation(x[0] * x[1] + x[2] * x[3])) \
        .add_var("h_next", ["C_next", "output_next"], lambda x: cell.kernel_activation(x[0]) * x[1])


LSTM_VARIANTS = {"NIG": lstm_nig_cell,
                 "NFG": lstm_nfg_cell,
                 "NOG": lstm_nog_cell,
                 "NIAF": lstm_niaf_cell,
                 "NOAF": lstm_noaf_cell,
                 "FGR": lstm_fgr_cell
                 }


class LSTMVariants(AbstractRNNBuilder):
    def __init__(self,
                 units: int,
                 lstm_cell: str,
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
        self.lstm_cell = lstm_cell

        super(LSTMVariants, self).__init__(units, kernel_activation, recurrent_activation,
                                           kernel_initializer, recurrent_initializer, bias_initializer,
                                           backward, return_sequences, use_bias, name="LSTM_%s" % lstm_cell, **kwargs)

    def get_cell(self, units: int,
                 kernel_activation: str,
                 recurrent_activation: str,
                 kernel_initializer: str,
                 recurrent_initializer: str,
                 bias_initializer: str,
                 use_bias: bool,
                 **kwargs) -> RNNCellBuilder:

        return LSTM_VARIANTS[self.lstm_cell](units, self.peephole,
                                             kernel_activation, recurrent_activation,
                                             kernel_initializer, recurrent_initializer, bias_initializer,
                                             use_bias, **kwargs)


class BiLSTMVariants(AbstractBiRNNBuilder):
    def __init__(self,
                 units: int,
                 lstm_cell: str,
                 peephole: bool = False,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 return_sequences: bool = False,
                 use_bias: bool = True,
                 **kwargs):

        self.peephole = peephole
        self.lstm_cell = lstm_cell

        super(BiLSTMVariants, self).__init__(units, kernel_activation, recurrent_activation,
                                             kernel_initializer, recurrent_initializer, bias_initializer,
                                             return_sequences, use_bias, name="BiLSTM_%s" % lstm_cell, **kwargs)

    def get_cell_forward(self, units: int,
                         kernel_activation: str,
                         recurrent_activation: str,
                         kernel_initializer: str,
                         recurrent_initializer: str,
                         bias_initializer: str,
                         use_bias: bool,
                         **kwargs) -> RNNCellBuilder:

        return LSTM_VARIANTS[self.lstm_cell](units, self.peephole,
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

        return LSTM_VARIANTS[self.lstm_cell](units, self.peephole,
                                             kernel_activation, recurrent_activation,
                                             kernel_initializer, recurrent_initializer, bias_initializer,
                                             use_bias, **kwargs)

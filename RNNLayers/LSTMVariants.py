from RNNLayers.RNNBuilder import RNNCellBuilder, AbstractRNNBuilder, AbstractBiRNNBuilder


# LSTM without Input Gate
def lstm_nig_cell(units: int,
                  peephole: bool = False,
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


